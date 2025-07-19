import cv2
import torch

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# 加载YOLOv5模型（local路径，强制刷新缓存）
model = torch.hub.load('F:/yolov5/yolov5-master/yolov5-master',
                       'custom',
                       path="F:/yolov5/yolov5-master/yolov5-master/runs/train/exp6/weights/best.pt",
                       source='local',
                       force_reload=True)
model.conf = 0.5  # 置信度阈值

# 打开摄像头
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取帧")
            break

        # 转换BGR到RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 模型推理
        results = model(img_rgb)

        # 解析检测结果
        detections = results.pandas().xyxy[0]

        for _, det in detections.iterrows():
            x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
            cls_name = det['name']
            conf = det['confidence']
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            label = f"{cls_name} {conf:.2f}"

            # 画框，点，标签
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Webcam YOLOv5 Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC键退出
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
