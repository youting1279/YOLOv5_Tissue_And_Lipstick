import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from torchvision import transforms

# YOLOv5 模型加载
# model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
# model = torch.hub.load(r'D:\boao\yolo\yolov5', 'custom', path=r'D:\boao\yolo\yolov5\runs\train\exp6\weights\best.pt',
#                        source='local')
# model = torch.hub.load('F:/yolov5/yolov5-master/yolov5-master','custom', path='yolov5s.pt',source='local')
model = torch.hub.load('F:/yolov5/yolov5-master/yolov5-master','custom',path="F:/yolov5/yolov5-master/yolov5-master/runs/train/exp4/weights/best.pt",source='local')

model.conf = 0.5  # 置信度阈值

# RealSense 相机初始化
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 启动相机
pipeline.start(config)

# 获取相机内参
profile = pipeline.get_active_profile()
color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
intrinsics = color_profile.get_intrinsics()

try:
    while True:
        # 等待帧
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # 转换图像
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        # YOLOv5 目标检测
        results = model(color_image)
        detections = results.pandas().xyxy[0]

        # 处理每个检测结果
        for _, det in detections.iterrows():
            x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
            cls_name = det['name']
            conf = det['confidence']

            # 计算边界框中心点
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # 获取深度值 (毫米)
            depth_value = depth_image[cy, cx]

            # 将像素坐标转换为3D坐标
            depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
            point_3d = rs.rs2_deproject_pixel_to_point(
                depth_intrin, [cx, cy], depth_value
            )

            # 转换为米
            point_3d = np.array(point_3d) / 1000.0  # 毫米转米
            # point_3d = np.array(point_3d)

            # 显示信息
            label = f"{cls_name} {conf:.2f}"
            position = f"({point_3d[0]:.2f}, {point_3d[1]:.2f}, {point_3d[2]:.2f})m"

            # 绘制边界框和文本
            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(color_image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(color_image, position, (x1, y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 显示结果
        cv2.imshow('RGB Detection', color_image)
        cv2.imshow('Depth', depth_colormap)

        # 按ESC退出
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # 停止和清理
    pipeline.stop()
    cv2.destroyAllWindows()