import cv2
import os

# path1=r'D:\boao\yolo\image'#生成的所有图片的路径
path1=r'F:\yolov5\yolov5-master\yolov5-master\images_tissue and lipstick'#生成的所有图片的路径
all_img_num=5#图片名字的位数，五位数例如：00001.jpg
def readtest():
    # videoname = r'D:\boao\yolo\视频\WIN_20250702_13_22_45_Pro.mp4'#视频的名字
    videoname = r'F:\yolov5\videos\Tissues and lipstick.mp4'
    capture = cv2.VideoCapture(videoname)
    if (not os.path.exists(path1)):
        os.makedirs(path1)
    num=0
    if capture.isOpened():
        while True:
            ret,img=capture.read() # img 就是一帧图片
            # cv.namedWindow("enhanced", 0);
            # cv.resizeWindow("enhanced", 640, 480);
            # cv2.imshow('enhanced',img)
            # cv2.waitKey(0)

            name=''
            for_name=''
            if len(str(num))<all_img_num:
                for i in range(len(str(num)),all_img_num):#对名字补零，保证图片名字命名格式一致
                    for_name+='0'
                name=for_name+str(num)
            else:
                name=str(num)
            name=path1+name
            name+='.jpg'
            print(name)
            cv2.imwrite(name, img)
            num+=1
            # 可以用 cv2.imshow() 查看这一帧，也可以逐帧保存
            if not ret:break # 当获取完最后一帧就结束
    else:
        print('视频打开失败！')

readtest()