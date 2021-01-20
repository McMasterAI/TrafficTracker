import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from detections import Load_Yolo_Model,yolo_predict
from yolov5.utils.datasets import letterbox

import time
import argparse

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort import generate_detections as gdet

# Helpers
def read_class_names(class_file_name):
    # loads class name from a file
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names
def efficiency_statistics(detection_times,tracking_times):
    ms = sum(detection_times)/len(detection_times)*1000
    fps = 1000 / ms
    fps2 = 1000 / (sum(tracking_times)/len(tracking_times)*1000)
    return ms,fps,fps2          
def get_tracker_info(tracker, val_list,key_list):
    tracked_bboxes = []
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 5:
            continue 
        bbox = track.to_tlbr() # Get the corrected/predicted bounding box
        class_name = track.get_class() #Get the class name of particular object
        tracking_id = track.track_id # Get the ID for the particular track
        index = key_list[val_list.index(class_name)] # Get predicted object index by object name
        tracked_bboxes.append(bbox.tolist() + [tracking_id, index]) # Structure data, that we could use it with our draw_bbox function
    return tracked_bboxes
def draw_image(original_frame,tracked_bboxes,CLASSES,fps=None):
    image = draw_bbox(original_frame, tracked_bboxes, CLASSES=CLASSES, tracking=True)
    if fps !=None:
        image = cv2.putText(image, "Time: {:.1f} FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
    return image
def get_video_capture_info(vid):
    # by default VideoCapture returns float instead of int
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    return width,height,fps

def Object_tracking(Yolo, video_path, output_path, CLASSES, input_size=416, show=False, score_threshold=0.3, iou_threshold=0.45, rectangle_colors='', Track_only = []):
    # Definition of the parameters
    max_cosine_distance = 0.7
    nn_budget = None
    
    #initialize deep sort object
    model_filename = 'models/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)


    if video_path:
        vid = cv2.VideoCapture(video_path) # detect on video
    else:
        vid = cv2.VideoCapture(0) # detect from webcam
    width, height, fps = get_video_capture_info(vid)
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, codec, fps, (width, height)) # output_path must be .mp4

    NUM_CLASS = read_class_names(CLASSES)
    key_list = list(NUM_CLASS.keys()) 
    val_list = list(NUM_CLASS.values())

    detection_times, tracking_times = [], []

    while True:
        _, frame = vid.read() #BGR

        # create the original_frame for display purposes (draw_bboxes)
        original_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)

        # preprocessing found in datasets.py
        img = letterbox(frame, new_shape=(height, width))[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        
        t1 = time.time()
        boxes, names, scores = yolo_predict(yolo,img, frame)
        t2 = time.time()

        features = np.array(encoder(original_frame, boxes))
        # Pass detections to the deepsort object and obtain the track information.
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(boxes, scores, names, features)]
        tracker.predict()
        tracker.update(detections)

        # Obtain info from the tracks
        tracked_bboxes = get_tracker_info(tracker, val_list,key_list)
        
        # update the times information
        t3 = time.time()
        detection_times.append(t2-t1)
        tracking_times.append(t3-t1)
        detection_times = detection_times[-20:]
        tracking_times = tracking_times[-20:]

        ms, fps, fps2 = efficiency_statistics(detection_times,tracking_times)

        # draw detection on frame
        image = draw_bbox(original_frame, tracked_bboxes, CLASSES=CLASSES, tracking=True)
        image = cv2.putText(image, "Time: {:.1f} FPS".format(fps), (0, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        # draw original yolo detection
        #image = draw_bbox(image, bboxes, CLASSES=CLASSES, show_label=False, rectangle_colors=rectangle_colors, tracking=True)

        # show and store the results
        print("Time: {:.2f}ms, Detection FPS: {:.1f}, total FPS: {:.1f}".format(ms, fps, fps2))
        if output_path != '': out.write(image)
        if show:
            cv2.imshow('output', image)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break   

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", help="path to input video",
        type=str, default = "inference/test.mp4")
    parser.add_argument("--output_path", help="where the outputs will be stored",
        type=str, default = "detection.mp4")
    parser.add_argument("--input_size", help="image input size",
        type=int, default = 416)
    parser.add_argument("--no_show", help="if mentioned, output images will not be shown, called without any argument",
        action="store_false",default = True)
    parser.add_argument("--iou_threshold", help="boolean for displaying output image",
        type=float, default = 0.1)
    parser.add_argument("--score_threshold", help="threshold for declaring a detection",
        type=float, default = 0.3)
    args = parser.parse_args()
    print('results')
    print(args.video_path, args.output_path, args.input_size,args.no_show,args.iou_threshold,args.score_threshold)
    yolo = Load_Yolo_Model()
    Object_tracking(yolo, args.video_path, args.output_path, CLASSES = "models/coco/coco.names",
        input_size=args.input_size, show=args.no_show, iou_threshold=args.iou_threshold,
        score_threshold=args.score_threshold, rectangle_colors=(255,0,0), Track_only = ["person"])
