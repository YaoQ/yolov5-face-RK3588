import cv2
import numpy as np
import argparse
import platform
from utils import COLORS, COCO_CLASSES
from rknnlite.api import RKNNLite
from postprocess_numpy import PostProcess

class YOLOv5:
    def __init__(self, args):
        # load bmodel
        self.rknn_lite = RKNNLite()
        ret = self.rknn_lite.load_rknn(args.model)
        if ret != 0:
            print('load rknnlite model failed!')
            exit(ret)

        self.rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
        self.batch_size = 1
        self.net_h = 640
        self.net_w = 640

        self.agnostic = True
        self.multi_label = True
        self.conf_thresh = args.conf_thresh
        self.nms_thresh = args.nms_thresh
        self.max_det = 1000
        
        self.postprocess = PostProcess(
            conf_thresh=self.conf_thresh,
            nms_thresh=self.nms_thresh,
            agnostic=self.agnostic,
            multi_label=self.multi_label,
            max_det=self.max_det,
        )
    
    def preprocess(self, ori_img):
        """
        pre-processing
        Args:
            img: numpy.ndarray -- (h,w,3)

        Returns: (3,h,w) numpy.ndarray after pre-processing

        """
        letterbox_img, ratio, (tx1, ty1) = self.letterbox(
            ori_img,
            new_shape=(self.net_h, self.net_w),
            color=(114, 114, 114),
            auto=False,
            scaleFill=False,
            scaleup=True,
            stride=32
        )
        img = letterbox_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        return img, ratio, (tx1, ty1) 
    
    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)
    
    def predict(self, input_img):
        return self.rknn_lite.inference(inputs=[input_img], data_format=['nchw']) 
    
    def __call__(self, img_list):
        img_num = len(img_list)
        ori_size_list = []
        preprocessed_img_list = []
        ratio_list = []
        txy_list = []
        for ori_img in img_list:
            ori_h, ori_w = ori_img.shape[:2]
            ori_size_list.append((ori_w, ori_h))
            preprocessed_img, ratio, (tx1, ty1) = self.preprocess(ori_img)
            preprocessed_img_list.append(preprocessed_img)
            ratio_list.append(ratio)
            txy_list.append([tx1, ty1])
        
        if img_num == self.batch_size:
            input_img = np.stack(preprocessed_img_list)
        else:
            input_img = np.zeros(self.input_shape, dtype='float32')
            input_img[:img_num] = np.stack(preprocessed_img_list)
        
            
        outputs = self.predict(input_img)
        results = self.postprocess(outputs, ori_size_list, ratio_list, txy_list)
        det = results[0]
        print(f"Find {det.shape[0]} faces")
        res_img = draw_numpy(img_list[0], det[:,:4], masks=None, classes_ids=det[:, 15], conf_scores=det[:, 4], landmarks=det[:, 5:15])
        cv2.imwrite("result.jpg", img_list[0])
        print("save result to result.jpg")

        return det

def draw_numpy(image, boxes, masks=None, classes_ids=None, conf_scores=None, landmarks=None):
    for idx in range(len(boxes)):
        x1, y1, x2, y2 = boxes[idx, :].astype(np.int32).tolist()
        #logging.debug("class id={}, score={}, (x1={},y1={},x2={},y2={})".format(classes_ids[idx],conf_scores[idx], x1, y1, x2, y2))
        if conf_scores[idx] < 0.25:
            continue
        if classes_ids is not None:
            color = COLORS[int(classes_ids[idx]) + 1]
        else:
            color = (0, 0, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness=1)
        if classes_ids is not None and conf_scores is not None:
            classes_ids = classes_ids.astype(np.int8)
            cv2.putText(image, "Face" + ':' + str(round(conf_scores[idx], 2)),
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=1)
        if masks is not None:
            mask = masks[:, :, idx]
            image[mask] = image[mask] * 0.5 + np.array(color) * 0.5

        
        clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
        for landmark in landmarks:
            for i in range(5):
                point_x = int(landmark[2 * i])
                point_y = int(landmark[2 * i + 1])
                cv2.circle(image, (point_x, point_y), 2, clors[i], -1)
        
    return image

def argsparser():
    parser = argparse.ArgumentParser(prog=__file__)
    parser.add_argument('--input', type=str, default='./img/face.jpg', help='path of input')
    parser.add_argument('--model', type=str, default='./weights/yolov5n-0.5.rknn', help='path of bmodel')
    parser.add_argument('--conf_thresh', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--nms_thresh', type=float, default=0.5, help='nms threshold')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = argsparser()
    yolov5_face = YOLOv5(args)
    result = yolov5_face([cv2.imread(args.input)])
