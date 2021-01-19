# 这个文件用来捕捉训练用的图片并保存的程序




import numpy as np
import time
import cv2

# 打开摄像头
camera = cv2.VideoCapture(0)
#camera.set(cv2.CAP_PROP_FRAME_WIDTH, 100)
#camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 100)

i = 0

# 遍历每一帧
while True:
    # 读取帧
    (ret, frame) = camera.read()
    fps = camera.get(cv2.CAP_PROP_FPS)
    # 判断是否成功打开摄像头
    if not ret:
        print
        'No Camera'
        break

    cv2.imshow('Frame', frame)
    i = i + 1
    print("i=", i)
    path = "E:\photo/" + str(i) + ".jpg"
    cv2.imwrite(path, frame)

    # 键盘检测，检测到esc键退出
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

# 摄像头释放
camera.release()
# 销毁所有窗口
cv2.destroyAllWindows()

