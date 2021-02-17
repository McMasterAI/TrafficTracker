from detections import Load_Yolo_Model, yolo_predict
import random
import colorsys
from deep_sort import generate_detections as gdet
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import nn_matching
import argparse
import time
from yolov5.utils.datasets import letterbox
import tensorflow as tf
import numpy as np
import cv2
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Temporary imports, associated with draw_bbox

# Helpers

def read_class_names(class_file_name, desired_classes=[]):
    # loads class name from a file
    names = {}
    desired_classes_names = [] # order of this list does not matter
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            name_stripped = name.strip('\n')
            if name_stripped in desired_classes:
                desired_classes_names.append(ID)
            names[ID] = name_stripped
    return names, desired_classes_names

def preprocess_image(img0, image_size):
    # preprocessing found in datasets.py
    img = letterbox(img0, new_shape=image_size)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    return img

def efficiency_statistics(detection_times, tracking_times):
    ms = sum(detection_times)/len(detection_times)*1000
    fps = 1000 / ms
    fps2 = 1000 / (sum(tracking_times)/len(tracking_times)*1000)
    return ms, fps, fps2


def get_tracker_info(tracker, val_list, key_list):
    tracked_bboxes = []
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 5:
            continue
        bbox = track.to_tlbr()  # Get the corrected/predicted bounding box
        class_name = track.get_class()  # Get the class name of particular object
        tracking_id = track.track_id  # Get the ID for the particular track
        # Get predicted object index by object name
        index = key_list[val_list.index(class_name)]
        # Structure data, that we could use it with our draw_bbox function
        tracked_bboxes.append(bbox.tolist() + [tracking_id, index])
    return tracked_bboxes

def get_video_capture_info(vid):
    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    return width, height, fps


def draw_bbox(image, bboxes, class_names, show_label=True, show_confidence=True, Text_colors=(255, 255, 0), rectangle_colors='', tracking=False):
    num_classes = len(class_names)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    #print("hsv_tuples", hsv_tuples)
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        if bbox_thick < 1:
            bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        (x1, y1), (x2, y2) = (coor[0], coor[1]), (coor[2], coor[3])

        # put object rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick*2)

        if show_label:
            # get text label
            score_str = " {:.2f}".format(score) if show_confidence else ""
            if tracking:
                score_str = " "+str(score)
            try:
                label = "{}".format(class_names[class_ind]) + score_str
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


def Object_tracking(Yolo, video_path, output_path, class_names, image_size=416, show=False,  rectangle_colors=''):
    # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None

    # initialize deep sort object
    model_filename = 'models/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    if video_path:
        vid = cv2.VideoCapture(video_path)  # detect on video
    else:
        vid = cv2.VideoCapture(0)  # detect from webcam
    width, height, fps = get_video_capture_info(vid)
    codec = cv2.VideoWriter_fourcc(*'XVID')
    # output_path must be .mp4
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))

    key_list = list(class_names.keys())
    val_list = list(class_names.values())

    detection_times, tracking_times = [], []
    _, frame = vid.read()  # BGR

    while frame is not None:
        # create the original_frame for display purposes (draw_bboxes)
        original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        # preprocessing found in datasets.py
        img = preprocess_image(frame, image_size)

        t1 = time.time()
        boxes, class_inds, scores = yolo_predict(yolo, img, frame)
        t2 = time.time()
        names = []
        for clss in class_inds:
            names.append(class_names[clss])
        features = np.array(encoder(original_frame, boxes))
        # Pass detections to the deepsort object and obtain the track information.
        detections = [Detection(bbox, score, class_name, feature) for bbox,
                      score, class_name, feature in zip(boxes, scores, names, features)]
        tracker.predict()
        tracker.update(detections)

        # Obtain info from the tracks
        tracked_bboxes = get_tracker_info(tracker, val_list, key_list)

        # update the times information
        t3 = time.time()
        detection_times.append(t2-t1)
        tracking_times.append(t3-t1)
        detection_times = detection_times[-20:]
        tracking_times = tracking_times[-20:]

        ms, fps, fps2 = efficiency_statistics(detection_times, tracking_times)

        # draw detection on frame
        image = draw_bbox(original_frame, tracked_bboxes,
                          class_names, tracking=True, rectangle_colors=rectangle_colors)
        image = cv2.putText(image, "Time: {:.1f} FPS".format(
            fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        
        # get next frame
        _, frame = vid.read()  # BGR


        # show and store the results
        print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(
            ms, fps, fps2))
        if output_path != '':
            out.write(image)
        if show:
            cv2.imshow('output', image)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    cv2.destroyAllWindows()


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
    parser.add_argument("--image_size", help="image input size",
                        type=int, default=640)
    parser.add_argument("--no_show", help="if mentioned, output images will not be shown, called without any argument",
                        action="store_false", default=True)
    parser.add_argument("--iou_threshold", help="boolean for displaying output image",
                        type=float, default=0.1)
    parser.add_argument("--conf_threshold", help="threshold for declaring a detection",
                        type=float, default=0.5)
    
    args = parser.parse_args()
    desired_classes = ['person', 'bicycle', 'car', 'motorbike', 'bus','truck']
    class_names, desired_classes_names = read_class_names(args.label_names_path, desired_classes=desired_classes)
    yolo = Load_Yolo_Model(conf_thres=args.conf_threshold,iou_thres=args.iou_threshold,imgsz = args.image_size, track_only=desired_classes_names, weights=args.weights_path)
    Object_tracking(yolo, args.video_path, args.output_path, class_names,
                    image_size=args.image_size, show=args.no_show)
