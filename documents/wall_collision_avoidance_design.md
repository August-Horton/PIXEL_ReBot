# 抓取过程中凸出墙壁碰撞规避算法设计文档

> 版本：v1.0  
> 适用对象：reBot-DevArm 六轴机械臂 + Realsense/Orbbec 深度相机（eye-in-hand）抓取流水线  
> 关联工程：`/home/enderplum/rebot_grasp`（基于 [set-cube-07261120.py](set-cube-07261120.py) 的抓取-放置流程）

---

## 1. 概述

### 1.1 目标

在机械臂从"预备位 → 抓取位 → 抬升 → 放置位"的行径过程中，实时感知环境中的**墙壁平面与凸出物**，规划并执行与所有障碍物保持安全距离的运动轨迹；当行径中检测到潜在碰撞风险时，动态调整抓取姿态与轨迹；在碰撞不可避免时触发应急保护。

### 1.2 设计约束（基于现有工程）

| 约束 | 说明 |
|---|---|
| 行径接口 | 现有 SDK 的 `RebotArmEndPose.move_to_traj(x,y,z,roll,pitch,yaw,duration)`，IK 失败返回 False |
| 感知硬件 | eye-in-hand 深度相机，`cam.get_frame()` 返回彩色图 + 深度图，`K` 为内参 |
| 坐标系统一 | 相机系 → 基座系：`T_cam2base = compose_cam_to_base_transform(tcp_pose, T_hand_eye, cfg)` |
| 目标定位 | 复用 YOLO 检测 + `estimate_grasps()` 得到目标位姿（相机系），变换到基座系 |
| 实时性目标 | 感知+检测周期 ≤ 100ms；路径规划单次 ≤ 50ms；动态监测周期 50ms |
| 现有能力复用 | `_visual_servo_center()`（视觉伺服居中）、`_blind_grasp()`、`safe_home()`、`_reset_wrist_joints()` |

### 1.3 名词约定

- **墙壁**：由 RANSAC 从点云拟合出的无限大平面，数学表示 `n·x + d = 0`
- **凸出物**：点云中超出墙面平面一定距离的点簇，建模为 AABB/OBB 包围盒
- **工具模型**：末端执行器（夹爪）的等效扫掠体，建模为球（半径 `tool_radius_m`）
- **安全距离**：工具球表面与障碍物表面的最小允许间距

---

## 2. 总体架构

```
┌────────────────────────────── 避障算法层（WallAvoidancePlanner） ─────────────────────────────┐
│                                                                                               │
│  ① 环境感知模块          ② 碰撞检测子系统          ③ 路径规划模块                              │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐                                 │
│  │ PointCloud    │      │ SceneModel   │      │ SafePath     │                                 │
│  │ 预处理/分割    │ ───▶ │ (墙/凸出物/   │ ───▶ │ Planner      │                                 │
│  │ RANSAC+聚类   │      │  目标/工具)   │      │ (直线/绕行/   │                                 │
│  └──────────────┘      └──────────────┘      │  多段检查)    │                                 │
│  ▲ 深度图+内参          ▲ 距离/碰撞预测        └──────────────┘                                 │
│                                                                                               │
│  ④ 动态调整机制          ⑥ 异常处理模块          ⑦ 评估与优化组件                               │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐                                 │
│  │ Monitor      │ ───▶ │ Emergency    │ ◀─── │ Evaluator    │                                 │
│  │ 速度外推/状态机│      │ 急停/报警/撤退 │      │ 合成场景/指标  │                                 │
│  └──────────────┘      └──────────────┘      └──────────────┘                                 │
│                                                                                               │
│  ⑤ 安全阈值设定：WallAvoidanceConfig（YAML 可配置）                                             │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────── 现有抓取流水线（应用层） ──────────────────────────┐
│  set-cube-07261120.py：_execute_grasp_sequence() → controller.move_to_traj() │
│  行径前：planner.update_scene() / plan_safe_path()                            │
│  行径中：planner.monitor_and_adjust() → 决定继续/调整/急停                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 数据流

1. 抓取触发 → 冻结帧（彩色+深度）→ 生成相机系点云
2. 感知模块：点云去噪/降采样 → RANSAC 提取墙面平面 → 剩余点欧氏聚类 → 凸出物 AABB
3. 目标定位：YOLO 检测框 + 深度 → `estimate_grasps()` → 基座系目标位姿（现有流程复用）
4. `T_cam2base` 将墙/凸出物/目标统一到基座系 → 更新 `SceneModel`
5. 路径规划：当前 TCP 位姿 → 目标位姿 → 直线检测 → 需要则生成绕行路点 → 输出 `SafePath`
6. 逐段调用 `move_to_traj` 执行；执行间隙调用动态监测
7. 监测触发 → 调整路点重规划或进入应急流程

---

## 3. 子系统详细设计

## 3.1 环境感知模块

**职责**：把深度相机原始数据转化为结构化环境模型（墙平面、凸出物、目标物体）。

### 3.1.1 点云生成

```
深度图 depth_mm (H,W) + 内参 K (fx,fy,cx,cy)
→ 反投影: X=(u-cx)/fx·Z, Y=(v-cy)/fy·Z, Z=depth/1000
→ 滤波: Z ∈ [min_depth, max_depth], 移除无效像素
→ 体素降采样 (voxel_size=0.01m)   # 降点数、去冗余
```

### 3.1.2 墙面平面提取（RANSAC）

迭代 ≤ 200 次，随机 3 点拟合平面，保留内点（距离 < `plane_thresh_m`）最多的平面为墙面；剔除墙面内点后剩余点视为凸出物候选点。墙平面参数 `(n, d)` 满足 `n·x + d = 0`。

### 3.1.3 凸出物聚类与建模

对候选点做**欧氏聚类**（体素网格 BFS，邻接阈值 `cluster_eps_m`），每簇生成 AABB 包围盒，再沿墙面法向做一次 OBB 化处理（提高大尺寸凸出物的紧凑度）。输出 `BoxObstacle{min, max, center, size, normal}`。

### 3.1.4 目标物体定位

复用现有 YOLO 检测 + `estimate_grasps()`，取有效抓取位姿（相机系）→ `T_cam2base` 变换 → 基座系 `target_pose`。若需感知到其他动态障碍（行人等），同一聚簇结果中与目标类别无关的大簇亦可登记为临时障碍。

### 3.1.5 输出

```python
@dataclass
class SceneModel:
    walls: list[WallPlane]        # 墙平面（基座系）
    obstacles: list[BoxObstacle]  # 凸出物 AABB（基座系）
    target: Pose6d | None         # 抓取目标位姿（基座系）
    timestamp: float              # 感知时间戳（用于数据新鲜度判断）
```

## 3.2 碰撞检测子系统

**职责**：基于 `SceneModel` 与工具模型，计算"工具-障碍物"距离并预测碰撞。

### 3.2.1 几何模型

| 对象 | 模型 | 数学表示 |
|---|---|---|
| 墙壁 | 半空间/平面 | `n·x + d = 0`，距离 = `|n·x + d|` |
| 凸出物 | AABB（OBB 可选） | 6 个半空间，距离取各轴钳制距离范数 |
| 工具（TCP 球） | 半径 `tool_radius_m` 的球 | 球心 = TCP 位置 |
| 运动路径 | 分段线段 | 起点→终点，按 `path_sampling_dt` 采样 |

### 3.2.2 距离计算

- `dist_point_to_plane(p, wall)`：点到平面距离
- `dist_point_to_aabb(p, box)`：钳制法 `q=clip(p,min,max); ||p-q||`，内部为 0
- `min_dist_to_path(seg, obstacle)`：对线段等距采样 ≥20 点取最小距离；关键段用解析最近点（线段-球/线段-AABB）
- **墙壁特殊处理**：TCP 必须始终处于墙的可达侧（`n·p + d > wall_safe_side`），穿越即判碰撞

### 3.2.3 碰撞预测

- **静态预测**：整条候选路径逐采样点检测，输出最小净距 `clearance` 与首碰位置
- **动态外推**：由当前 TCP 速度 `v` 外推未来 `prediction_horizon_s` 秒位置，检查外推线段

### 3.2.4 输出

```python
@dataclass
class CollisionReport:
    min_clearance: float          # 最小净距（>0 安全）
    collided: bool
    hit_point: np.ndarray | None  # 首碰点
    obstacle: BoxObstacle | None  # 涉事障碍物
    samples_checked: int
```

## 3.3 路径规划模块

**职责**：在碰撞检测基础上生成"起点 → 目标"的安全分段路径。

### 3.3.1 规划策略（分层）

1. **L0 直线优先**：直接对起点→目标直线做碰撞检测；全部安全 → 单段直行（与现状零开销一致）
2. **L1 绕行**：检测到碰撞区间 `[s_start, s_end]` 后，构造绕行路点：
   - **抬升绕行**（默认）：碰撞区段中点上方 `lift_offset_m` 处插入抬升路点，另在区段两端插入上/下坡路点，形成 `起→升→越→降→止` 五段路径
   - **侧向绕行**（墙法向受限时）：沿水平面上与"墙法向"垂直的方向偏移 `clearance_m + protrusion_width/2`
   - 组合候选：`[抬升, 侧偏, 抬升+侧偏]` 依次尝试
3. **L2 关节空间备选**：绕行失败时，调用 SDK `solve_ik` 获取多解，选择"关节构型远离障碍物"的解（目标函数含障碍物关节位置代价），再由 `plan_joint_space_trajectory` 执行
4. **L3 失败 → 交异常处理模块**：返回 `None`，触发应急

### 3.3.2 路径校验

对每个候选路径逐段采样做全量碰撞检测，`clearance` 不低于 `path_min_clearance_m` 才接受；递归加深绕行高度/偏移最多 `max_retry=3` 次，避免死循环。

### 3.3.3 输出

```python
@dataclass
class SafePath:
    waypoints: list[Pose6d]       # 依次执行的路点（含起点终点）
    clearance: float              # 全程最小净距
    strategy: str                 # 'direct' | 'lift' | 'side' | 'joint'
    segment_check: list[float]    # 每段最小净距
```

执行时对每段调用 `move_to_traj`（原 SDK 接口，无需改动底层）。

## 3.4 动态调整机制

**职责**：行径执行中持续监测，风险出现时实时调整姿态与轨迹。

### 3.4.1 监测循环

每 `check_period_s`（默认 50ms）执行一次：

```
tcp_pose_now, tcp_vel
外推线段 = [tcp_pos, tcp_pos + tcp_vel·prediction_horizon_s]
new_cloud → update_scene()（增量：只对凸出物部分做帧间匹配，静止物体不变）
clearance = min 距离(外推线段 vs 障碍物 ∪ 墙)
```

### 3.4.2 三级状态机

| 状态 | 判定 | 动作 |
|---|---|---|
| NORMAL | `clearance ≥ warning_dist` | 继续执行 |
| ADJUSTING | `warning_dist > clearance ≥ critical_dist` | 计算避让方向（沿障碍物最近点法向向外），在剩余路径前方插入修正路点并重规划（轻量：仅重规划剩余段）；同时可微调末端姿态（pitch/yaw）避免夹爪棱角 |
| CRITICAL | `clearance < critical_dist` | 转入异常处理模块 |

调整方向计算：`dir = (tcp_pos - nearest_obstacle_point)` 归一化，抬升分量优先（沿 +Z 可同时满足墙/桌面约束），再叠加水平分量。

### 3.4.3 与视觉伺服协同

重规划期间可暂时冻结视觉伺服的 PID 修正（参照 `_visual_servo_center` 的死区/盲走逻辑），避免"避障调整"与"居中修正"相互打架；避障路点到位后再恢复伺服。

## 3.5 安全阈值设定

所有阈值集中在 YAML 配置节 `wall_avoidance:`，运行时载入 `WallAvoidanceConfig`。

```yaml
wall_avoidance:
  enable: true
  # --- 感知 ---
  min_depth_m: 0.05
  max_depth_m: 1.0
  voxel_size_m: 0.01
  plane_thresh_m: 0.02        # RANSAC 平面内点阈值
  cluster_eps_m: 0.03         # 凸出物聚类邻接阈值
  # --- 碰撞检测 ---
  tool_radius_m: 0.04         # 工具等效球半径（夹爪外扩）
  wall_safe_side_m: 0.03      # 墙可达侧最小穿透裕量
  path_sampling_dt_s: 0.02    # 路径采样时间步（配合速度→空间分辨率）
  # --- 路径规划 ---
  path_min_clearance_m: 0.05  # 路径允许最小净距（与墙/凸出物）
  protrusion_clearance_m: 0.06# 凸出物避让距离（L1 绕行目标净距）
  lift_offset_m: 0.10         # 抬升绕行高度
  side_offset_m: 0.12         # 侧向绕行偏移量
  max_retry: 3
  # --- 动态调整 ---
  check_period_s: 0.05
  prediction_horizon_s: 0.5
  warning_dist_m: 0.08        # 预警阈值
  critical_dist_m: 0.03       # 急停阈值
  # --- 异常处理 ---
  emergency_action: "retreat" # stop | alarm | retreat | safe_home
  retreat_speed: 0.2
  alarm_sound: true
```

参数标定方法：`warning_dist` ≥ 单周期最大位移 + 工具半径 + 感知误差；`critical_dist` ≥ 急停制动距离 + 工具半径。

## 3.6 异常处理模块

**职责**：碰撞不可避免时保证机械臂安全。

### 3.6.1 触发条件

- 动态监测进入 CRITICAL
- 路径规划 L0~L3 全部失败
- IK 连续失败 / 感知数据超时（`timestamp` 过期 > `data_timeout_s`）

### 3.6.2 应急流程（按 `emergency_action` 配置）

1. **立即暂停**：锁定 `controller._q_target` 不变（或停止 `_send_thread` 下发），中断当前运动
2. **告警**：`alarm_sound` + 控制台/日志输出障碍物位置与最近距离（供后续排查）
3. **撤退**（默认 `retreat`）：沿**安全路径栈**反向逐路点返回起点（路径规划时记录每段安全路点）；若栈为空则执行 `controller.safe_home()`（关节最小 jerk 回零，速度限制 `retreat_speed`）
4. **状态复位**：清空规划缓存与状态机，等待人工确认后恢复

### 3.6.3 接口

```python
class EmergencyHandler:
    def handle(self, reason: str, report: CollisionReport | None) -> None
    # 内部: pause_motion() → alarm() → retreat()/safe_home()
```

## 3.7 算法评估与优化组件

**职责**：量化避碰效果，驱动参数与算法迭代。

### 3.7.1 离线仿真评估（自测入口 `self_test()`）

- 合成场景生成器：随机墙平面方位（法向偏角 ±30°）、凸出物（位置/长宽高/数量 1~4）、目标点（墙前 0.2~0.6m），生成 ≥200 个场景
- 对每场景运行完整规划流程，统计：
  - **避碰成功率**：规划出 `clearance ≥ path_min_clearance_m` 的路径占比
  - **最小净距分布**：均值/最差（反映安全裕量）
  - **路径冗余度**：`实际路径长度 / 直线距离`（反映绕行效率）
  - **计算耗时**：规划耗时 p50/p95（实时性）
- 输出结构化报告（dict），可落盘 CSV 供回归对比

### 3.7.2 实际应用数据采集

每次抓取记录：场景快照（墙/凸出物参数）、规划策略、全程逐点最小净距序列、是否触发调整/应急。落盘 `wall_avoid_log.jsonl`。

### 3.7.3 优化回路

- 敏感性分析：扫描 `path_min_clearance_m` / `lift_offset_m` 对成功率与冗余度的影响曲线，选择 Pareto 最优点
- 碰撞检测热点优化：对 RANSAC 与聚类阶段做 numpy 向量化；距离计算对障碍物建立空间索引（体素哈希）降低复杂度
- 回归保障：每次参数/算法变更后重跑合成场景集，成功率不得回退

---

## 4. 接口定义汇总

```python
class WallAvoidancePlanner:
    def __init__(self, cfg: WallAvoidanceConfig): ...
    # ① 感知
    def update_scene_from_cloud(self, cloud: np.ndarray, T_cam2base: np.ndarray) -> None
    def update_scene_manual(self, walls=..., obstacles=..., target=...) -> None  # 调试/离线用
    # ② 碰撞检测
    def check_clearance(self, seg_start, seg_end) -> CollisionReport
    # ③ 路径规划
    def plan_safe_path(self, start: Pose6d, target: Pose6d) -> SafePath | None
    # ④ 动态调整
    def monitor_and_adjust(self, tcp_pose, tcp_vel, cloud=None) -> Advisory  # NORMAL/ADJUSTING/CRITICAL
    # ⑥ 异常处理
    def handle_emergency(self, reason: str) -> None
    # ⑦ 评估
    def run_evaluation(self, n_scenes: int) -> dict
```

## 5. 与现有工程集成方案

集成位置：[set-cube-07261120.py](set-cube-07261120.py) 的 `_execute_grasp_sequence()`（以及放置旋转段可选）。

```python
# 初始化（main 中一次）
planner = WallAvoidancePlanner(WallAvoidanceConfig.from_dict(cfg.get("wall_avoidance", {})))

# 抓取前（_execute_grasp_sequence Step1 后）
cloud = build_cloud(snap_depth, K)                     # 深度→点云（相机系）
planner.update_scene_from_cloud(cloud, T_cam2base)     # ① 感知

# 行径：预抓取位/抓取位 两段
ok, path = planner.plan_safe_path(tcp_now, pregrasp6d) # ③ 规划
if path is None:
    planner.handle_emergency("plan_failed"); return False
for wp in path.waypoints:                              # 逐段执行
    if not controller.move_to_traj(*wp, duration=...):
        return False
    adv = planner.monitor_and_adjust(tcp_pose, vel)    # ④ 动态监测
    if adv.level == "CRITICAL":
        planner.handle_emergency("critical"); return False
    if adv.level == "ADJUSTING":
        replan_and_execute(planner, controller, adv)   # 重规划剩余段
```

详细接入示例见 `set-cube-07261120-wallavoid.py`。

## 6. 测试与验证计划

| 层级 | 内容 | 通过标准 |
|---|---|---|
| 单元 | 距离计算（点到平面/AABB/线段）数值正确性 | 与解析解误差 < 1e-6 |
| 单元 | 绕行路点生成（抬升/侧偏）几何合法性 | 路点位于安全侧 |
| 集成 | 合成场景 ≥200 组规划 | 成功率 ≥ 95%，p95 规划 < 50ms |
| 半实物 | 点云来自真实相机（墙+纸箱场景） | 墙平面法向误差 < 5° |
| 实物 | 完整抓取-放置流程带凸出障碍 | 0 碰撞，成功放置 |

## 7. 风险与边界

- **感知盲区**：eye-in-hand 相机在行进中视角受限，凸出物可能在镜头外 → 依赖 L1 绕行高度余量与静态场景假设；建议增加行进前多角度快扫
- **点云噪声/反光**：玻璃/金属面 → 增加深度滤波与聚类尺寸下限过滤
- **动态障碍**：当前版本假设墙壁/凸出物静止；行人等动态障碍需接入 3.1.4 的临时障碍注册并缩短预测周期
- **夹爪棱角**：工具球模型忽略形状细节 → `tool_radius_m` 应取夹爪最大外接球半径并加 20% 裕量
- **计算资源**：RANSAC+聚类在低算力板卡需降采样 + 帧差静态化（静止障碍只更新一次）
