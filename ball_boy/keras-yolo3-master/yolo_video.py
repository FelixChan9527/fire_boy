import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import numpy as np
import time
from timeit import default_timer as timer
import cv2

#python E:/dection/keras-yolo3-master/yolo_video.py --image
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 192*5)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 108*5)

def detect_img(yolo):
    i = 0
    while True:
        star = timer()  # 刚开始循环时间

        # 读取帧
        (ret, frame) = camera.read()
        # 判断是否成功打开摄像头
        if not ret:
            print
            'No Camera'
            break
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) #opencv格式转PIL格式
        r_image = yolo.detect_image(image)      #识别图像
        # r_image.show()
        # r_image.close()
        show_image = cv2.cvtColor(np.asarray(r_image), cv2.COLOR_RGB2BGR)      #将识别后输出PIL图像转怕opencv图像
        # cv2.namedWindow('show_image', 0)
        # cv2.resizeWindow('show_image', 192*5, 108*5)
        cv2.imshow('show_image', show_image)

        # 键盘检测，检测到esc键退出
        k = cv2.waitKey(1) & 0xFF
        if (k == 27) | (i == 1000):
            break

        # 结束循环时间
        end = timer()
        fps = 1 / (end - star)  # 计算帧率
        print("fps = ", fps)

    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
