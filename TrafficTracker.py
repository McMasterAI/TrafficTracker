import colorsys
import csv
import time
from threading import Thread

import numpy as np

from config_parser import get_config
from deep_sort import build_tracker
from detections import *
from yolov5.utils.datasets import letterbox

deep_sort_path = "models/deep_sort.yaml"
video_path = "inference/test_2fps.mp4"
image_size = 416
label_names_path = "models/coco/coco.names"
columns = ["created_time", "Pos_x", "Pos_y", "width",
           "height", "Class", "Object_id", "location_id"]


class TrafficTracker(Thread):
    def __init__(self):
        cfg = get_config()
        cfg.merge_from_file(deep_sort_path)
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            print("Using CPU")

        desired_classes = ['person', 'bicycle',
                           'car', 'motorbike', 'bus', 'truck']
        class_names, desired_class_names = read_class_names(
            label_names_path, desired_classes=desired_classes)
        self.class_names = class_names

        self.yolo = Load_Yolo_Model(track_only=desired_class_names)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        print("Initialized!")

    def run(self, video_path, start_time=time.time()):
        frame_time = start_time

        self.get_new_video_capture(video_path)
        _, frame = self.vid.read()  # BGR

        metrics = []
        while frame is not None:
            img = self.preprocess_image(frame, image_size)

            boxes, class_inds, scores = yolo_predict(self.yolo, img, frame)
            boxes = np.array([list(box) for box in boxes])
            names = [self.class_names[name] for name in class_inds]
            outputs = self.deepsort.update(boxes, names, scores, frame)

            # Generate Metrics
            frame_time = self.get_next_time(frame_time, self.vid_fps)

            for i, _ in enumerate(outputs):
                metrics.append({
                    "created_time": frame_time,
                    "Pos_x": outputs[i][0],
                    "Pos_y": outputs[i][1],
                    "width": outputs[i][2],
                    "height": outputs[i][3],
                    "Class": outputs[i][5],
                    "Object_id": outputs[i][4],
                    "location_id": "McMaster University"
                })

            if len(outputs) > 0:
                # draw bboxes
                pass

            _, frame = self.vid.read()  # BGR

        save_csv(columns, metrics)

    def get_new_video_capture(self, video_path=video_path):
        self.vid = cv2.VideoCapture(video_path)
        self.vid_width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vid_height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vid_fps = int(self.vid.get(cv2.CAP_PROP_FPS))

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

        for _, bbox in enumerate(bboxes):
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

    @staticmethod
    def initialize_video_writer(output_path, fps, width, height):
        codec = cv2.VideoWriter_fourcc(*'XVID')
        # output_path must be .mp4
        out = cv2.VideoWriter(output_path, codec, fps, (width, height))
        return codec, out

    @staticmethod
    def preprocess_image(img0, image_size):
        # preprocessing found in datasets.py
        img = letterbox(img0, new_shape=image_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        return img

    @staticmethod
    def xywh_to_xyxy(coor):
        w, h = coor[2], coor[3]
        x1, y1 = max(coor[0]-w//2, 0), max(coor[1]-h//2, 0)
        x2, y2 = x1 + w, y1 + h
        return x1, y1, x2, y2

    @staticmethod
    def get_next_time(time, fps):
        return time + 1.0/fps


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


def save_csv(columns, dict_data):
    with open("traffic_data.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for data in dict_data:
            writer.writerow(data)


if __name__ == "__main__":
    traffic = TrafficTracker()
    traffic.run("inference/test_2fps.mp4")
