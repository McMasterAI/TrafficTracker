from detections import Load_Yolo_Model, yolo_predict
import random
import colorsys
from deep_sort import build_tracker
from config_parser import get_config
import torch
import argparse
import time
from yolov5.utils.datasets import letterbox
import tensorflow as tf
import numpy as np
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Temporary imports, associated with draw_bbox

class TrafficTracker:
    def __init__(self, options):
        self.video_path = options['video_path']
        self.output_path = options['output_path']
        self.class_names = options['class_names']
        self.image_size = options['imgsz']
        self.show = options['show']
        self.deep_sort_path = options['deep_sort_path']
        self.iou_thres = options['iou_thres']
        self.conf_thres = options['conf_thres']
        self.track_only = options['track_only']
        self.weights_path = options['weights_path']

        self.rectangle_colors = ''
        self.max_cosine_distance = 0.7
        self.nn_budget = None
        self.key_list = list(self.class_names.keys())
        self.val_list = list(self.class_names.values())

    def initialize_deep_sort(self):
        # initialize deep sort object
        cfg = get_config()
        cfg.merge_from_file(self.deep_sort_path)
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            print("Running in cpu mode which maybe very slow!", UserWarning)
        return build_tracker(cfg, use_cuda=use_cuda)

    def get_video_capture_info(self, vid):
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        return width, height, fps

    def initialize_video_capture(self, video_path):
        if video_path:
            vid = cv2.VideoCapture(self.video_path)  # detect on video
        else:
            vid = cv2.VideoCapture(0)  # detect from webcam
        return vid

    def initialize_video_writer(self, output_path, fps, width, height):
        codec = cv2.VideoWriter_fourcc(*'XVID')
        # output_path must be .mp4
        out = cv2.VideoWriter(output_path, codec, fps, (width, height))
        return codec, out

    def initialize_models(self):
        self.yolo = Load_Yolo_Model(conf_thres=self.conf_thres,
                                    iou_thres=self.iou_thres, imgsz=self.image_size, track_only=self.track_only, weights=self.weights_path)    
        self.tracker = self.initialize_deep_sort()
        self.vid = self.initialize_video_capture(self.video_path)
        self.width, self.height, self.fps = self.get_video_capture_info(
            self.vid)
        self.codec, self.out = self.initialize_video_writer(
            self.output_path, self.fps, self.width, self.height)


    def preprocess_image(self, img0, image_size):
        # preprocessing found in datasets.py
        img = letterbox(img0, new_shape=image_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return img

    def efficiency_statistics(self, detection_times, tracking_times):
        ms = sum(detection_times)/len(detection_times)*1000
        fps = 1000 / ms
        fps2 = 1000 / (sum(tracking_times)/len(tracking_times)*1000)
        return ms, fps, fps2

    def xywh_to_xyxy(self, coor):
        w, h = coor[2], coor[3]
        x1, y1 = max(coor[0]-w//2, 0), max(coor[1]-h//2, 0)
        x2, y2 = x1 + w, y1 + h
        return x1, y1, x2, y2

    def draw_bbox(self, image, bboxes, class_names, show_label=True, show_confidence=True, Text_colors=(255, 255, 0), rectangle_colors='', tracking=False):
        num_classes = len(class_names)
        image_h, image_w, _ = image.shape
        hsv_tuples = [(1.0 * x / num_classes, 1., 1.)
                      for x in range(num_classes)]
        #print("hsv_tuples", hsv_tuples)
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        for i, bbox in enumerate(bboxes):
            coor = np.array(bbox[:4], dtype=np.int32)
            score = int(bbox[4])
            class_name = bbox[5]
            bbox_color = rectangle_colors
            bbox_thick = int(0.6 * (image_h + image_w) / 1000)
            if bbox_thick < 1:
                bbox_thick = 1
            fontScale = 0.75 * bbox_thick
            x1, y1, x2, y2 = self.xywh_to_xyxy(coor)
            # put object rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

            if show_label:
                # get text label
                score_str = " {:.2f}".format(score) if show_confidence else ""
                if tracking:
                    score_str = " "+str(score)
                try:
                    label = "{}".format(class_name) + score_str
                except KeyError:
                    print(
                        "You received KeyError, this might be that you are trying to use yolo original weights")
                    print(
                        "while using custom classes, if using custom model in configs.py set YOLO_CUSTOM_WEIGHTS = True")
                # get text size
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                                      fontScale, thickness=bbox_thick)
                # put filled text rectangle
                cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 -
                                                text_height - baseline), bbox_color, thickness=cv2.FILLED)
                # put text above rectangle
                cv2.putText(image, label, (x1, y1-4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)
        return image

    def run(self):
        detection_times, tracking_times = [], []
        _, frame = self.vid.read()  # BGR

        while frame is not None:
            img = self.preprocess_image(frame, self.image_size)

            t1 = time.time()
            boxes, class_inds, scores = yolo_predict(self.yolo, img, frame, None)
            t2 = time.time()
            names = []
            for clss in class_inds:
                names.append(self.class_names[clss])
            # Pass detections to the deepsort object and obtain the track information.

            # this should be done in yolo_predict!
            boxes = np.array([list(box) for box in boxes])
            # The image offset problem is I think that last parameter should be img instead of frame/original_frame
            tracked_bboxes = self.tracker.update(boxes, names, scores, frame)

            # update the times information
            t3 = time.time()
            detection_times.append(t2-t1)
            tracking_times.append(t3-t1)
            detection_times = detection_times[-20:]
            tracking_times = tracking_times[-20:]
            ms, fps, fps2 = self.efficiency_statistics(
                detection_times, tracking_times)

            # get next frame
            _, frame = self.vid.read()  # BGR

            # show and store the results
            print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(
                ms, fps, fps2))
            if self.show:
                # draw detection on frame
                original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                image = self.draw_bbox(original_frame, tracked_bboxes,
                                       class_names, tracking=True, rectangle_colors=self.rectangle_colors)
                image = cv2.putText(image, "Time: {:.1f} FPS".format(
                    fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
                cv2.imshow('output', image)
                if self.output_path != '':
                    self.out.write(image)
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break

    cv2.destroyAllWindows()


def read_class_names(class_file_name, desired_classes=[]):
    # loads class name from a file
    names = {}
    desired_classes_names = []  # order of this list does not matter
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            name_stripped = name.strip('\n')
            if name_stripped in desired_classes:
                desired_classes_names.append(ID)
            names[ID] = name_stripped
    return names, desired_classes_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", help="path to input video",
                        type=str, default="inference/test.mp4")
    parser.add_argument("--output_path", help="where the outputs will be stored",
                        type=str, default="detection.mp4")
    parser.add_argument("--label_names_path", help="path to the names of the labels",
                        type=str, default="models/coco/coco.names")
    parser.add_argument('--weights_path', nargs='+', type=str,
                        default='models/yolov5s.pt', help='path to weights, __model__.pt, path')
    parser.add_argument('--deep_sort_path', nargs='+', type=str,
                        default='models/deep_sort.yaml', help='path to weights, __model__.pt, path')
    parser.add_argument("--image_size", help="image input size",
                        type=int, default=640)
    parser.add_argument("--no_show", help="if mentioned, output images will not be shown, called without any argument",
                        action="store_false", default=True)
    parser.add_argument("--iou_threshold", help="boolean for displaying output image",
                        type=float, default=0.1)
    parser.add_argument("--conf_threshold", help="threshold for declaring a detection",
                        type=float, default=0.5)

    args = parser.parse_args()
    desired_classes = ['person', 'bicycle', 'car', 'motorbike', 'bus', 'truck']
    class_names, desired_classes_names = read_class_names(
        args.label_names_path, desired_classes=desired_classes)
    options = {
        'conf_thres': args.conf_threshold,
        'iou_thres': args.iou_threshold,
        'imgsz': args.image_size,
        'track_only': desired_classes_names,
        'weights_path': args.weights_path,
        'video_path': args.video_path,
        'output_path': args.output_path,
        'deep_sort_path': args.deep_sort_path,
        'class_names': class_names,
        'show': args.no_show
    }

    traffic_tracker = TrafficTracker(options)
    traffic_tracker.initialize_models()
    traffic_tracker.run()
