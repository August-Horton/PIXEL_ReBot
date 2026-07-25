from ultralytics import YOLO
import torch

print(torch.cuda.is_available())  # 必须返回 True
print(torch.cuda.get_device_name(0)) # 应该显示你的显卡型号
# 加载模型
model = YOLO('yolo11n.pt')

# 推理并显示
# show=True 会自动调用 OpenCV 弹出窗口实时显示
results = model.predict(source=0, show=True, conf=0.5,device=0)
