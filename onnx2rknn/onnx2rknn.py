
import cv2
import numpy as np
import argparse

from rknn.api import RKNN
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", '--onnx', type=str, default='onnx_model/yolov5n-0.5.onnx', help='weights path')  
    parser.add_argument("-n", '--name', type=str, default='yolov5s', help='model name') 
    parser.add_argument('--img_size', nargs='+', type=int, default=[640,640], help='inference size h,w')
    parser.add_argument('--rknn', type=str, default='./rknn_models', help='保存路径')
    parser.add_argument('--platform', type=str, default='rk3588')
    parser.add_argument('--mean_values',  type=list, default=[[0., 0., 0.]])
    parser.add_argument('--scale_values', type=list, default=[[255.0, 255.0, 255.0]])
    parser.add_argument('--datasets', type=str, default="dataset.txt", help="Datasets path file for calibration")
    args = parser.parse_args()


    platform = args.platform
    Height, Width = args.img_size
    MODEL_PATH = args.onnx
    exp = os.path.basename(args.onnx).split('.')[0]
    NEED_BUILD_MODEL = True

    # Create RKNN object
    rknn = RKNN()

    OUT_DIR = args.rknn
    RKNN_MODEL_PATH = './{}/{}.rknn'.format(
        OUT_DIR, exp+'-'+str(Width)+'-'+str(Height), platform)
    if NEED_BUILD_MODEL:
        DATASET = args.datasets
        rknn.config(mean_values=args.mean_values, std_values=args.scale_values, target_platform=platform
                    )
        # Load model
        print('--> Loading model')
        ret = rknn.load_onnx(MODEL_PATH)
        if ret != 0:
            print('load model failed!')
            exit(ret)
        print('done')

        # Build model
        print('--> Building model')
        ret = rknn.build(do_quantization=True, dataset=DATASET)
        if ret != 0:
            print('build model failed.')
            exit(ret)
        print('done')

        # Export rknn model
        if not os.path.exists(OUT_DIR):
            os.mkdir(OUT_DIR)
        print('--> Export RKNN model: {}'.format(RKNN_MODEL_PATH))
        ret = rknn.export_rknn(RKNN_MODEL_PATH)
        if ret != 0:
            print('Export rknn model failed.')
            exit(ret)
        print('done')

        ret = rknn.accuracy_analysis(inputs=['../img/face.jpg'])
    else:
        ret = rknn.load_rknn(RKNN_MODEL_PATH)

    rknn.release()

