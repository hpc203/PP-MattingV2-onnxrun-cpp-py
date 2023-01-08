#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse
import cv2
import numpy as np
import onnxruntime

class PP_MattingV2:
    def __init__(self, modelpath, conf_thres=0.65):
        self.conf_threshold = conf_thres
        # Initialize model
        self.onnx_session = onnxruntime.InferenceSession(modelpath)
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_name = self.onnx_session.get_outputs()[0].name

        self.input_shape = self.onnx_session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
    def prepare_input(self, image):
        input_image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), dsize=(self.input_width, self.input_height))
        input_image = input_image.astype(np.float32) / 255.0
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        return input_image

    def detect(self, image):
        input_image = self.prepare_input(image)

        # Perform inference on the image
        result = self.onnx_session.run([self.output_name], {self.input_name: input_image})

        # Post process:squeeze
        segmentation_map = result[0]
        segmentation_map = np.squeeze(segmentation_map)

        image_width, image_height = image.shape[1], image.shape[0]
        dst_image = copy.deepcopy(image)
        segmentation_map = cv2.resize(
            segmentation_map,
            dsize=(image_width, image_height),
            interpolation=cv2.INTER_LINEAR,
        )

        # color list
        color_image_list = []
        # ID 0:BackGround
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (0, 0, 0)
        color_image_list.append(bg_image)
        # ID 1:Human
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (0, 255, 0)
        color_image_list.append(bg_image)

        mask = np.where(segmentation_map > self.conf_threshold, 0, 1)
        mask = np.stack((mask,) * 3, axis=-1).astype('uint8')
        mask_image = np.where(mask, dst_image, color_image_list[1])
        dst_image = cv2.addWeighted(dst_image, 0.5, mask_image, 0.5, 1.0)
        return dst_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='images/3.jpg', help="image path")
    parser.add_argument('--modelpath', type=str, default='weights/ppmattingv2_stdc1_human_480x640.onnx', help="image path")
    parser.add_argument('--confThreshold', default=0.65, type=float, help='class confidence')
    parser.add_argument('--use_video', type=int, default=0, help="if use video")
    parser.add_argument('--videopath', type=str, default='images/3.mp4', help="video path")
    args = parser.parse_args()

    # Initialize YOLOv7 object detector
    segmentor = PP_MattingV2(args.modelpath, conf_thres=args.confThreshold)
    if args.use_video != 1:
        srcimg = cv2.imread(args.imgpath)

        # Detect Objects
        dstimg = segmentor.detect(srcimg)
        winName = 'PP-MattingV2 in ONNXRuntime'
        cv2.namedWindow(winName, 0)
        cv2.imshow(winName, dstimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        # cap = cv2.VideoCapture(0)  ###电脑的摄像头
        cap = cv2.VideoCapture(args.videopath)  ###视频文件
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            dstimg = segmentor.detect(frame)
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
            cv2.imshow('PP-MattingV2 in ONNXRuntime', dstimg)

        cap.release()
        cv2.destroyAllWindows()