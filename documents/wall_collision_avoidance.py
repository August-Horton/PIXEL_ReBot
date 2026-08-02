"""抓取过程中凸出墙壁碰撞规避算法模块。

独立实现, 不依赖现有工程与硬件, 使用纯 numpy 可离线自测。
对应设计文档: wall_collision_avoidance_design.md

七大子系统:
  ① 环境感知模块   update_scene_from_cloud() / update_scene_manual()
  ② 碰撞检测子系统 check_clearance() / 几何距离计算
  ③ 路径规划模块   plan_safe_path()  (直线优先 → 抬升/侧偏绕行 → 校验)
  ④ 动态调整机制   monitor_and_adjust()  (速度外推 + 三级状态机)
  ⑤ 安全阈值设定   WallAvoidanceConfig (dataclass + dict 加载)
  ⑥ 异常处理模块   handle_emergency()  (暂停/告警/撤退/回零回调)
  ⑦ 评估与优化     run_evaluation() / self_test()

坐标约定:
  - 所有几何量均为基座系 (机器人基座), 单位米
  - 墙壁平面: n·x + d = 0, n 为单位法向量, 指向墙的可达侧
  - 凸出物: 轴对齐包围盒 AABB {min, max}
  - 工具: 以 TCP 为球心的等效球, 半径 tool_radius_m

运行自测:
    python3 wall_collision_avoidance.py
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

# ===========================================================================
# ⑤ 安全阈值设定
# ===========================================================================


@dataclass
class WallAvoidanceConfig:
    """避障安全阈值与算法参数 (对应 YAML 节 wall_avoidance)。"""

    # --- 感知 ---
    min_depth_m: float = 0.05
    max_depth_m: float = 1.0
    voxel_size_m: float = 0.01          # 体素降采样
    plane_thresh_m: float = 0.02        # RANSAC 平面内点阈值
    cluster_eps_m: float = 0.03         # 凸出物欧氏聚类邻接阈值
    # --- 碰撞检测 ---
    tool_radius_m: float = 0.04         # 工具等效球半径
    wall_safe_side_m: float = 0.03      # 墙可达侧最小穿透裕量
    path_sampling_dt_s: float = 0.02    # 路径采样时间步
    # --- 路径规划 ---
    path_min_clearance_m: float = 0.05  # 路径允许最小净距
    protrusion_clearance_m: float = 0.06  # 凸出物避让目标净距
    lift_offset_m: float = 0.10         # 抬升绕行高度
    side_offset_m: float = 0.12         # 侧向绕行偏移量
    max_retry: int = 3
    # --- 动态调整 ---
    check_period_s: float = 0.05
    prediction_horizon_s: float = 0.5
    warning_dist_m: float = 0.08        # 预警阈值
    critical_dist_m: float = 0.03       # 急停阈值
    # --- 异常处理 ---
    emergency_action: str = "retreat"   # stop | alarm | retreat | safe_home
    retreat_speed: float = 0.2

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "WallAvoidanceConfig":
        """从配置 dict (YAML 解析结果) 构建, 未知键忽略。"""
        if not data:
            return cls()
        known = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        return cls(**kwargs)


# ===========================================================================
# 数据结构
# ===========================================================================


@dataclass
class WallPlane:
    """墙壁平面: n·x + d = 0, n 为单位法向量指向可达侧。"""

    normal: np.ndarray  # (3,) 单位向量
    d: float            # 常数项

    def signed_dist(self, p: np.ndarray) -> float:
        """有符号距离: 正=可达侧, 负=墙后 (穿越即碰撞)。"""
        return float(np.dot(self.normal, p) + self.d)


@dataclass
class BoxObstacle:
    """凸出物 AABB 包围盒 (基座系)。"""

    min: np.ndarray     # (3,)
    max: np.ndarray     # (3,)
    label: str = "protrusion"

    @property
    def center(self) -> np.ndarray:
        return (self.min + self.max) * 0.5

    @property
    def size(self) -> np.ndarray:
        return self.max - self.min

    def dist_point(self, p: np.ndarray) -> float:
        """点到 AABB 距离, 内部为 0。"""
        q = np.clip(p, self.min, self.max)
        return float(np.linalg.norm(p - q))

    def nearest_point(self, p: np.ndarray) -> np.ndarray:
        """AABB 上离 p 最近的点。"""
        return np.clip(p, self.min, self.max)


@dataclass
class SceneModel:
    """环境模型 (基座系)。"""

    walls: list[WallPlane] = field(default_factory=list)
    obstacles: list[BoxObstacle] = field(default_factory=list)
    target: Optional[np.ndarray] = None   # 抓取目标位置 (3,)
    timestamp: float = 0.0

    def empty(self) -> bool:
        return not self.walls and not self.obstacles


@dataclass
class CollisionReport:
    """碰撞检测结果。"""

    min_clearance: float = float("inf")   # 最小净距
    collided: bool = False
    hit_point: Optional[np.ndarray] = None
    obstacle: Optional[BoxObstacle] = None
    samples_checked: int = 0


@dataclass
class Pose6d:
    """末端位姿 (位置 + RPY 欧拉角)。"""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    def as_tuple(self) -> tuple[float, ...]:
        return (self.x, self.y, self.z, self.roll, self.pitch, self.yaw)

    @classmethod
    def from_array(cls, xyz: np.ndarray, rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> "Pose6d":
        return cls(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]),
                   roll=rpy[0], pitch=rpy[1], yaw=rpy[2])


@dataclass
class SafePath:
    """规划出的安全路径。"""

    waypoints: list[Pose6d]       # 依次执行的路点 (含起点/终点)
    clearance: float              # 全程最小净距
    strategy: str = "direct"      # direct | lift | side | lift+side
    segment_clearance: list[float] = field(default_factory=list)

    @property
    def length(self) -> float:
        if len(self.waypoints) < 2:
            return 0.0
        pts = np.array([w.pos for w in self.waypoints])
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


@dataclass
class Advisory:
    """动态监测输出。"""

    level: str = "NORMAL"          # NORMAL | ADJUSTING | CRITICAL
    clearance: float = float("inf")
    avoid_dir: Optional[np.ndarray] = None   # 建议避让方向 (单位向量)
    nearest_obstacle: Optional[BoxObstacle] = None
    nearest_point: Optional[np.ndarray] = None


# ===========================================================================
# ① 环境感知模块
# ===========================================================================


def depth_to_cloud(depth_mm: np.ndarray, K: np.ndarray,
                   min_z: float, max_z: float) -> np.ndarray:
    """深度图 (H,W,mm) + 内参 K[[fx,0,cx],[0,fy,cy],[0,0,1]] → 相机系点云 (N,3) m。"""
    h, w = depth_mm.shape[:2]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    z = depth_mm.astype(np.float64) / 1000.0
    valid = (z > min_z) & (z < max_z)
    z = z[valid]
    u = u[valid]
    v = v[valid]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """体素降采样, 每体素保留一个代表点。"""
    if points.shape[0] == 0:
        return points
    vox = np.floor(points / max(voxel_size, 1e-6)).astype(np.int64)
    _, idx = np.unique(vox, axis=0, return_index=True)
    return points[idx]


def fit_plane_ransac(points: np.ndarray, thresh: float, iters: int = 200,
                     seed: int = 0) -> tuple[Optional[WallPlane], np.ndarray]:
    """RANSAC 拟合最大平面。

    返回 (WallPlane | None, 内点索引)。内点满足 |n·x + d| <= thresh。
    n 的方向随机取向: 调用方应按"可达侧"约定翻转。
    """
    n = points.shape[0]
    if n < 3:
        return None, np.zeros(0, dtype=bool)
    rng = np.random.default_rng(seed)
    best: Optional[WallPlane] = None
    best_mask: Optional[np.ndarray] = None
    best_count = 0
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        a, b, c = points[idx]
        normal = np.cross(b - a, c - a)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        d = -float(np.dot(normal, a))
        dist = np.abs(points @ normal + d)
        mask = dist <= thresh
        count = int(mask.sum())
        if count > best_count:
            # 用内点精化一次最小二乘平面
            inliers = points[mask]
            centroid = inliers.mean(axis=0)
            _, _, vh = np.linalg.svd(inliers - centroid)
            refined = vh[-1]
            if refined[2] < 0:
                refined = -refined
            d_ref = -float(np.dot(refined, centroid))
            mask_ref = np.abs(points @ refined + d_ref) <= thresh
            best_count = int(mask_ref.sum())
            best = WallPlane(normal=refined, d=d_ref)
            best_mask = mask_ref
    if best is None:
        return None, np.zeros(0, dtype=bool)
    return best, best_mask


def cluster_obstacles(points: np.ndarray, eps: float, min_points: int = 20
                      ) -> list[BoxObstacle]:
    """体素网格 BFS 欧氏聚类 → AABB 列表。"""
    if points.shape[0] == 0:
        return []
    key = np.floor(points / max(eps, 1e-6)).astype(np.int64)
    # 用 dict 建立体素→点索引映射
    vox_map: dict[tuple[int, int, int], list[int]] = {}
    for i, k in enumerate(key):
        vox_map.setdefault(tuple(k), []).append(i)
    unvisited = set(range(points.shape[0]))
    clusters: list[list[int]] = []
    neighbors = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    while unvisited:
        start = unvisited.pop()
        queue = [start]
        cluster: list[int] = [start]
        while queue:
            cur = queue.pop()
            kc = tuple(key[cur])
            for dx, dy, dz in neighbors:
                nk = (kc[0] + dx, kc[1] + dy, kc[2] + dz)
                ids = vox_map.get(nk)
                if not ids:
                    continue
                for pid in ids:
                    if pid in unvisited:
                        unvisited.discard(pid)
                        queue.append(pid)
                        cluster.append(pid)
        if len(cluster) >= min_points:
            clusters.append(cluster)
    obstacles = []
    for cl in clusters:
        pts = points[cl]
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        obstacles.append(BoxObstacle(min=lo, max=hi))
    return obstacles


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """点云 (N,3) 应用 4x4 齐次变换 → (N,3)。"""
    if points.shape[0] == 0:
        return points
    hom = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    return (T @ hom.T).T[:, :3]


# ===========================================================================
# ② 碰撞检测子系统
# ===========================================================================


def segment_min_distance(a: np.ndarray, b: np.ndarray, n_samples: int = 20
                         ) -> tuple[np.ndarray, float]:
    """线段 [a,b] 等距采样 n_samples 个点。

    返回 (采样点 (n,3), 采样间距)。采样间距供调用方估算空间分辨率。
    """
    ts = np.linspace(0.0, 1.0, n_samples)
    pts = a[None, :] + ts[:, None] * (b - a)[None, :]
    return pts, float(np.linalg.norm(b - a)) / (n_samples - 1)


def check_segment_vs_obstacles(seg: tuple[np.ndarray, np.ndarray],
                               scene: SceneModel,
                               cfg: WallAvoidanceConfig,
                               n_samples: int = 20) -> CollisionReport:
    """检测线段与场景 (墙+凸出物) 的最近净距与碰撞。

    净距 = 点距 - tool_radius_m; 墙还要保证在可达侧 (signed_dist >= wall_safe_side)。
    """
    a, b = seg
    pts, _ = segment_min_distance(a, b, n_samples)
    report = CollisionReport(min_clearance=float("inf"), samples_checked=len(pts))

    def _consider(clearance: float, hit: np.ndarray, obs: Any) -> None:
        nonlocal report
        if clearance < report.min_clearance:
            report.min_clearance = clearance
            report.hit_point = hit.copy()
            report.obstacle = obs
            report.collided = clearance < 0.0

    for p in pts:
        for wall in scene.walls:
            sd = wall.signed_dist(p)
            # 可达侧正数裕量, 墙后 (sd < wall_safe_side) 视为碰撞
            _consider(sd - cfg.tool_radius_m - cfg.wall_safe_side_m, p, wall)
        for box in scene.obstacles:
            nearest = box.nearest_point(p)
            d = float(np.linalg.norm(p - nearest))
            _consider(d - cfg.tool_radius_m, nearest, box)
    return report


# ===========================================================================
# ③ 路径规划模块
# ===========================================================================


class SafePathPlanner:
    """分层路径规划: L0 直线 → L1 抬升/侧偏绕行 → L2 失败返回 None。"""

    def __init__(self, cfg: WallAvoidanceConfig):
        self.cfg = cfg

    def _path_clearance(self, waypoints: list[Pose6d]) -> float:
        """多段路径的最小净距。"""
        best = float("inf")
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            rep = check_segment_vs_obstacles(
                (a.pos, b.pos), self._scene, self.cfg)
            best = min(best, rep.min_clearance)
        return best

    def plan(self, scene: SceneModel, start: Pose6d, target: Pose6d) -> Optional[SafePath]:
        self._scene = scene
        # ---- L0: 直线 ----
        rep = check_segment_vs_obstacles((start.pos, target.pos), scene, self.cfg)
        if rep.min_clearance >= self.cfg.path_min_clearance_m:
            return SafePath(waypoints=[start, target], clearance=rep.min_clearance,
                            strategy="direct", segment_clearance=[rep.min_clearance])

        # ---- L1: 绕行 ----
        for retry in range(1, self.cfg.max_retry + 1):
            lift = self.cfg.lift_offset_m * retry
            side = self.cfg.side_offset_m * retry
            candidates = self._build_detours(start, target, scene, lift, side)
            for wp, name in candidates:
                clr = self._path_clearance(wp)
                if clr >= self.cfg.path_min_clearance_m:
                    return SafePath(waypoints=wp, clearance=clr, strategy=name)
        return None

    def _build_detours(self, start: Pose6d, target: Pose6d, scene: SceneModel,
                       lift: float, side: float
                       ) -> list[tuple[list[Pose6d], str]]:
        """构造绕行路点候选: 抬升 / 侧偏 / 抬升+侧偏。"""
        s, t = start.pos, target.pos
        mid = (s + t) * 0.5
        z_base = max(s[2], t[2])  # 抬升基准取较高端
        d = t - s
        horiz = np.array([d[0], d[1], 0.0])
        hlen = float(np.linalg.norm(horiz))
        # 侧向轴: 水平面上与行进方向垂直
        side_axis = np.array([-horiz[1], horiz[0], 0.0]) / max(hlen, 1e-6)
        # 若墙存在, 侧偏方向取与墙法向垂直的横向
        if scene.walls:
            n = scene.walls[0].normal
            side_axis = np.cross(n, np.array([0.0, 0.0, 1.0]))
            side_axis = side_axis / max(float(np.linalg.norm(side_axis)), 1e-6)
        if float(np.linalg.norm(side_axis)) < 1e-6:  # 退化为任意横向
            side_axis = np.array([1.0, 0.0, 0.0])

        lift_pts = [Pose6d.from_array(s),
                    Pose6d.from_array(s + np.array([0, 0, lift])),
                    Pose6d.from_array(np.array([mid[0], mid[1], max(z_base, mid[2]) + lift])),
                    Pose6d.from_array(t + np.array([0, 0, lift])),
                    Pose6d.from_array(t)]
        side_pts = [Pose6d.from_array(s),
                    Pose6d.from_array(s + side_axis * side),
                    Pose6d.from_array(t + side_axis * side),
                    Pose6d.from_array(t)]
        combo_pts = [Pose6d.from_array(s),
                     Pose6d.from_array(s + side_axis * side + np.array([0, 0, lift])),
                     Pose6d.from_array(t + side_axis * side + np.array([0, 0, lift])),
                     Pose6d.from_array(t)]
        return [(lift_pts, "lift"), (side_pts, "side"), (combo_pts, "lift+side")]


# ===========================================================================
# ④ 动态调整机制
# ===========================================================================


def monitor_trajectory(scene: SceneModel, cfg: WallAvoidanceConfig,
                       tcp_pos: np.ndarray, tcp_vel: np.ndarray) -> Advisory:
    """速度外推 + 三级状态判定 (NORMAL / ADJUSTING / CRITICAL)。"""
    future = tcp_pos + tcp_vel * cfg.prediction_horizon_s
    rep = check_segment_vs_obstacles((tcp_pos, future), scene, cfg, n_samples=15)
    adv = Advisory(clearance=rep.min_clearance)
    if rep.obstacle is not None and rep.hit_point is not None:
        adv.nearest_obstacle = rep.obstacle if isinstance(rep.obstacle, BoxObstacle) else None
        adv.nearest_point = rep.hit_point
        delta = tcp_pos - rep.hit_point
        norm = float(np.linalg.norm(delta))
        if norm > 1e-6:
            adv.avoid_dir = delta / norm
    if rep.min_clearance < cfg.critical_dist_m:
        adv.level = "CRITICAL"
    elif rep.min_clearance < cfg.warning_dist_m:
        adv.level = "ADJUSTING"
    else:
        adv.level = "NORMAL"
    return adv


# ===========================================================================
# ⑥ 异常处理模块
# ===========================================================================


class EmergencyHandler:
    """应急处理: 暂停 → 告警 → 撤退/回零。

    与真实硬件解耦: 通过回调注入 pause/retreat/alarm 行为。
    """

    def __init__(self, cfg: WallAvoidanceConfig,
                 on_pause: Optional[Callable[[], None]] = None,
                 on_alarm: Optional[Callable[[str], None]] = None,
                 on_retreat: Optional[Callable[[], None]] = None):
        self.cfg = cfg
        self._on_pause = on_pause or (lambda: None)
        self._on_alarm = on_alarm or (lambda msg: print(f"[Emergency] ALARM: {msg}"))
        self._on_retreat = on_retreat or (lambda: None)
        self.triggered_count = 0

    def handle(self, reason: str, report: Optional[CollisionReport] = None) -> None:
        self.triggered_count += 1
        print(f"[Emergency] reason={reason} count={self.triggered_count}")
        if report is not None:
            print(f"[Emergency]   min_clearance={report.min_clearance:.4f}m "
                  f"hit={None if report.hit_point is None else report.hit_point.round(3)}")
        # 1. 立即暂停运动
        self._on_pause()
        # 2. 告警
        self._on_alarm(reason)
        # 3. 按配置撤退
        action = self.cfg.emergency_action
        if action in ("retreat", "safe_home"):
            print(f"[Emergency]   action={action} (speed={self.cfg.retreat_speed})")
            self._on_retreat()


# ===========================================================================
# ⑦ 算法评估与优化组件
# ===========================================================================


def make_synthetic_scene(seed: int) -> tuple[SceneModel, Pose6d, Pose6d]:
    """生成合成场景: 随机墙方位 + 1~4 个凸出物 + 起点/目标。"""
    rng = np.random.default_rng(seed)
    yaw = rng.uniform(-np.pi / 6, np.pi / 6)
    nx, ny = math.sin(yaw), math.cos(yaw)
    wall = WallPlane(normal=np.array([nx, ny, 0.0]), d=-0.5)  # x·n - 0.5 = 0

    obstacles = []
    n_obs = int(rng.integers(1, 5))
    for _ in range(n_obs):
        w = rng.uniform(0.05, 0.20)
        h = rng.uniform(0.05, 0.20)
        dep = rng.uniform(0.02, 0.10)
        base = np.array([rng.uniform(0.15, 0.35), rng.uniform(-0.25, 0.25), 0.0])
        obstacles.append(BoxObstacle(min=base, max=base + np.array([w, h, dep])))

    start = Pose6d.from_array(np.array([0.25, 0.0, 0.30]))
    target = Pose6d.from_array(np.array([0.35, 0.0, 0.08]))
    scene = SceneModel(walls=[wall], obstacles=obstacles, timestamp=time.time())
    return scene, start, target


class Evaluator:
    """离线仿真评估: 避碰成功率 / 最小净距 / 路径冗余度 / 规划耗时。"""

    def __init__(self, cfg: WallAvoidanceConfig):
        self.cfg = cfg

    def run(self, n_scenes: int = 200, seed: int = 42) -> dict:
        planner = SafePathPlanner(self.cfg)
        ok, fail = 0, 0
        clearances: list[float] = []
        redundancies: list[float] = []
        times: list[float] = []
        strategies: dict[str, int] = {}
        for i in range(n_scenes):
            scene, start, target = make_synthetic_scene(seed + i)
            t0 = time.perf_counter()
            path = planner.plan(scene, start, target)
            times.append((time.perf_counter() - t0) * 1000.0)
            if path is None:
                fail += 1
                continue
            ok += 1
            clearances.append(path.clearance)
            straight = float(np.linalg.norm(target.pos - start.pos))
            redundancies.append(path.length / max(straight, 1e-6))
            strategies[path.strategy] = strategies.get(path.strategy, 0) + 1
        return {
            "n_scenes": n_scenes,
            "success": ok,
            "success_rate": ok / max(n_scenes, 1),
            "min_clearance_avg_m": float(np.mean(clearances)) if clearances else float("nan"),
            "min_clearance_worst_m": float(np.min(clearances)) if clearances else float("nan"),
            "redundancy_avg": float(np.mean(redundancies)) if redundancies else float("nan"),
            "plan_ms_p50": float(np.percentile(times, 50)) if times else float("nan"),
            "plan_ms_p95": float(np.percentile(times, 95)) if times else float("nan"),
            "strategies": strategies,
        }


# ===========================================================================
# 顶层门面类
# ===========================================================================


class WallAvoidancePlanner:
    """避障算法统一入口, 对应设计文档接口定义汇总。"""

    def __init__(self, cfg: Optional[WallAvoidanceConfig] = None):
        self.cfg = cfg or WallAvoidanceConfig()
        self.scene = SceneModel(timestamp=time.time())
        self._path_planner = SafePathPlanner(self.cfg)
        self.emergency = EmergencyHandler(self.cfg)

    # ① 感知
    def update_scene_from_cloud(self, cloud_cam: np.ndarray,
                                T_cam2base: np.ndarray,
                                target_cam: Optional[np.ndarray] = None) -> SceneModel:
        """相机系点云 → 基座系场景模型。

        流程: 降采样 → RANSAC 墙面 → 剩余点聚类 → 凸出物 AABB。
        """
        pts = voxel_downsample(cloud_cam, self.cfg.voxel_size_m)
        pts = transform_points(pts, T_cam2base)
        wall, inlier_mask = fit_plane_ransac(pts, self.cfg.plane_thresh_m)
        walls: list[WallPlane] = []
        if wall is not None:
            walls.append(wall)
        rest = pts[~inlier_mask] if inlier_mask is not None else pts
        obstacles = cluster_obstacles(rest, self.cfg.cluster_eps_m)
        target = None
        if target_cam is not None:
            target = transform_points(target_cam.reshape(1, 3), T_cam2base)[0]
        self.scene = SceneModel(walls=walls, obstacles=obstacles,
                                target=target, timestamp=time.time())
        return self.scene

    def update_scene_manual(self, walls: Optional[list[WallPlane]] = None,
                            obstacles: Optional[list[BoxObstacle]] = None,
                            target: Optional[np.ndarray] = None) -> None:
        """手动/离线注入场景 (调试用)。"""
        self.scene = SceneModel(walls=walls or [], obstacles=obstacles or [],
                                target=target, timestamp=time.time())

    # ② 碰撞检测
    def check_clearance(self, seg_start: np.ndarray, seg_end: np.ndarray
                        ) -> CollisionReport:
        return check_segment_vs_obstacles((seg_start, seg_end), self.scene, self.cfg)

    # ③ 路径规划
    def plan_safe_path(self, start: Pose6d, target: Pose6d) -> Optional[SafePath]:
        path = self._path_planner.plan(self.scene, start, target)
        return path

    # ④ 动态调整
    def monitor_and_adjust(self, tcp_pos: np.ndarray, tcp_vel: np.ndarray) -> Advisory:
        return monitor_trajectory(self.scene, self.cfg, tcp_pos, tcp_vel)

    # ⑥ 异常处理
    def handle_emergency(self, reason: str, report: Optional[CollisionReport] = None) -> None:
        self.emergency.handle(reason, report)

    # ⑦ 评估
    def run_evaluation(self, n_scenes: int = 200) -> dict:
        return Evaluator(self.cfg).run(n_scenes)


# ===========================================================================
# 自测入口
# ===========================================================================


def self_test() -> int:
    """无硬件自测: 几何正确性 + 规划成功率 + 动态监测 + 应急。"""
    print("=" * 70)
    print("WallAvoidancePlanner self_test")
    print("=" * 70)

    # ---- 1. 距离计算正确性 ----
    box = BoxObstacle(min=np.array([0.0, 0.0, 0.0]), max=np.array([0.1, 0.1, 0.1]))
    assert abs(box.dist_point(np.array([0.2, 0.0, 0.0])) - 0.1) < 1e-9, "AABB 距离错误"
    assert box.dist_point(np.array([0.05, 0.05, 0.05])) == 0.0, "AABB 内部应距离 0"
    wall = WallPlane(normal=np.array([1.0, 0.0, 0.0]), d=-0.5)
    assert abs(wall.signed_dist(np.array([0.6, 0, 0])) - 0.1) < 1e-9, "平面距离错误"
    print("[1] 几何距离计算 OK")

    # ---- 2. 感知: 合成点云 → 墙 + 凸出物 ----
    rng = np.random.default_rng(0)
    wall_pts = rng.normal(0.0, 0.005, (3000, 3))
    wall_pts[:, 0] = 0.4
    bump_pts = rng.uniform([0.2, -0.1, 0.05], [0.3, 0.1, 0.12], (500, 3))
    cloud = np.vstack([wall_pts, bump_pts])
    T_identity = np.eye(4)
    planner = WallAvoidancePlanner()
    scene = planner.update_scene_from_cloud(cloud, T_identity)
    assert len(scene.walls) >= 1, "未提取到墙面"
    assert len(scene.obstacles) >= 1, "未提取到凸出物"
    print(f"[2] 感知 OK: 墙={len(scene.walls)} 凸出物={len(scene.obstacles)} "
          f"w0={scene.walls[0].normal.round(3)}")

    # ---- 3. 路径规划: 直线被凸出物阻挡 → 绕行 ----
    start = Pose6d.from_array(np.array([0.0, 0.0, 0.10]))
    target = Pose6d.from_array(np.array([0.5, 0.0, 0.10]))  # 穿越凸出物
    path = planner.plan_safe_path(start, target)
    assert path is not None and path.strategy != "direct", "应触发绕行"
    assert path.clearance >= planner.cfg.path_min_clearance_m - 1e-9
    print(f"[3] 路径规划 OK: strategy={path.strategy} clearance={path.clearance:.3f}m "
          f"路点数={len(path.waypoints)}")

    # 无障碍时直线
    planner2 = WallAvoidancePlanner()
    planner2.update_scene_manual(obstacles=[], walls=[])
    p2 = planner2.plan_safe_path(start, target)
    assert p2 is not None and p2.strategy == "direct"
    print(f"[4] 无障碍直线路径 OK: clearance={p2.clearance:.3f}m")

    # ---- 5. 动态监测 ----
    adv = planner.monitor_and_adjust(target.pos, np.array([0.2, 0.0, 0.0]))
    print(f"[5] 动态监测 OK: level={adv.level} clearance={adv.clearance:.3f}m")
    adv_far = planner2.monitor_and_adjust(np.array([0.0, 0.0, 0.4]), np.zeros(3))
    assert adv_far.level == "NORMAL"
    print(f"[6] 监测 NORMAL OK: clearance={adv_far.clearance:.3f}m")

    # ---- 7. 异常处理 ----
    calls: list[str] = []
    handler = EmergencyHandler(
        planner.cfg,
        on_pause=lambda: calls.append("pause"),
        on_alarm=lambda msg: calls.append("alarm"),
        on_retreat=lambda: calls.append("retreat"),
    )
    handler.handle("test_critical", None)
    assert calls == ["pause", "alarm", "retreat"], calls
    print("[7] 异常处理 OK:", calls)

    # ---- 8. 评估 ----
    report = planner.run_evaluation(n_scenes=120)
    print("[8] 评估报告:")
    for k, v in report.items():
        if isinstance(v, float):
            print(f"    {k:22s} = {v:.4f}")
        else:
            print(f"    {k:22s} = {v}")
    print("=" * 70)
    print("self_test PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_test())
