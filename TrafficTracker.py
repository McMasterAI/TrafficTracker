import argparse
import csv
import time
from threading import Thread

import numpy as np

from config_parser import get_config
from deep_sort import build_tracker
from detections import *
from yolov5.utils.datasets import letterbox

from tqdm import tqdm
import logging
from logutils import TqdmLoggingHandler


#from app.database_connector import insert_to_table

columns = ["created_time", "Pos_x", "Pos_y", "width",
           "height", "Class", "Object_id", "location_id"]
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

class TrafficTracker(Thread):
    def __init__(self):
        cfg = get_config()
        cfg.merge_from_file(opt.deep_sort_path)
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            print("Using CPU")

        desired_classes = ['person', 'bicycle',
                           'car', 'motorbike', 'bus', 'truck']
        class_names, desired_class_names = read_class_names(
            opt.label_names_path, desired_classes=desired_classes)
        self.class_names = class_names

        self.yolo = Load_Yolo_Model(track_only=desired_class_names,conf_thres=opt.confidence, weights=opt.yolo_path)
        self.deepsort = build_tracker(cfg, use_cuda=use_cuda)
        self.log = logging.getLogger()
        self.log.setLevel(logging.INFO)
        self.log.addHandler(TqdmLoggingHandler())
        print("Initialized!")

    def run(self, video_path, start_time=time.time()):
        frame_time = start_time

        self.get_new_video_capture(video_path)
        _, out = self.initialize_video_writer(
            opt.output_path, self.vid_fps, self.vid_width, self.vid_height)
        _, og_frame = self.vid.read()  # BGR
        
        metrics = []

        pbar = tqdm(total=self.length+1)

                
        while og_frame is not None:
            pbar.update()
            new_frame = self.preprocess_image(og_frame, opt.image_size)

            boxes, class_inds, scores = yolo_predict(
                self.yolo, new_frame, og_frame,self.log) # outputs boxes in tlwh format
            # checks if there is an output, otherwise just keeps the original frame
            if boxes and class_inds and scores:
                boxes = np.array([list(box) for box in boxes])
                names = [self.class_names[name] for name in class_inds]
                outputs = self.deepsort.update(boxes, names, scores, og_frame)

                # Generate Metrics
                frame_time = self.get_next_time(frame_time, self.vid_fps)

                for i, _ in enumerate(outputs):
                    #outputs are in format tlwh
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
                    bbox_tlwh = outputs[:, :4]
                    identities = outputs[:, 4]
                    classes = [i[-1] for i in outputs]
                    og_frame = self.draw_boxes(og_frame, bbox_tlwh, classes  , identities)
                    out.write(og_frame)

            _, og_frame = self.vid.read()  # BGR

        self.vid.release()
        out.release()
        save_csv(columns, metrics, opt.csv_path)
        #insert_to_table("dbo.heatmap", opt.csv_path)

    def get_new_video_capture(self, video_path):
        self.vid = cv2.VideoCapture(video_path)
        self.vid_width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.vid_height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.vid_fps = int(self.vid.get(cv2.CAP_PROP_FPS))
        self.length = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))


    def draw_boxes(self, img, bbox, classes, identities=None):
        for i, box in enumerate(bbox):
            x, y, w, h = [int(i) for i in box]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0
            cl = classes[i]
            color = self.compute_color_for_labels(id)
            label = '{}{:d}  {}'.format("", id,cl)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 3)
            cv2.rectangle(
                img, (x, y), (x+t_size[0]+3, y+t_size[1]+4), color, -1)
            cv2.putText(
                img, label, (x, y+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        return img

    @staticmethod
    def compute_color_for_labels(label):
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        return tuple(color)

    @staticmethod
    def initialize_video_writer(output_path, fps, width, height):
        codec = cv2.VideoWriter_fourcc(*'H264') # output_path must be .mp4
        out = cv2.VideoWriter(output_path, codec, fps, (width, height))
        return codec, out

    @staticmethod
    def preprocess_image(img0, image_size):
        # preprocessing found in datasets.py
        img = letterbox(img0, new_shape=image_size)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)
        return img

    @staticmethod
    def xywh_to_tlwh(coor):
        w, h = coor[2], coor[3]
        x = max(int(coor[0])-int(w)//2, 0)
        y = max(int(coor[1])-int(h)//2, 0)
        return x, y, w, h

    @staticmethod
    def xyxy_to_tlwh(bbox_xyxy):
        x1,y1,x2,y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x1-x2)
        h = int(y1-y2)
        return t,l,w,h

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


def save_csv(columns, dict_data, csv_path):
    with open(csv_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for data in dict_data:
            writer.writerow(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--deep_sort_path', nargs='+', type=str,
                        default='models/deep_sort.yaml', help='deep_sort YAML path')
    parser.add_argument('--yolo_path', nargs='+', type=str,
                        default='models/yolov5s.pt', help='model.pt path')
    parser.add_argument('--video_path', type=str,
                        default='inference/test_10fps.mp4', help='source video file path')
    parser.add_argument('--output_path', type=str,
                        default='inference/output.mp4', help='output video file path')
    parser.add_argument('--label_names_path', type=str,
                        default='models/coco/coco.names', help='label enumerations path')
    parser.add_argument('--csv_path', type=str,
                        default='traffic_data.csv', help='save path for output csv')
    parser.add_argument('--image_size', type=int,
                        default=640, help='height for the resolution to downscale video frames to')
    parser.add_argument('--confidence', type=float,
                        default=0.25, help='Confidence threshold for yolo classifier as a percentage from 0 to 1')
    opt = parser.parse_args()

    traffic = TrafficTracker()
    traffic.run(opt.video_path)
