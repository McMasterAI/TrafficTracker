import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random, zeros_like

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (check_img_size, non_max_suppression,
                                  scale_coords, set_logging, strip_optimizer,
                                  xyxy2xywh)
from yolov5.utils.torch_utils import select_device, time_synchronized

# The two lines below, import sys AND sys.path.insert(0,'./yolov5') , are necessary in order
# to avoid modules not found errors.
# https://github.com/ultralytics/yolov5/issues/353
sys.path.insert(0, './yolov5')


def Load_Yolo_Model(device=select_device(''), conf_thres=0.51, iou_thres=0.45, weights='models/yolov5s.pt', 
    imgsz=640, track_only=[]):
    """Save a yolo model object.
    Args:
        device (device): Which device to use, cpu or gpu.
        weights (String): Relative location of weights file.
        track_only (String list): list of object classes to track.
    Returns:
        model: yolov5 model.
    """
    yolov5 = attempt_load(weights, map_location=device)
    yolov5.device = device
    yolov5.weights = weights
    # half precision only supported on CUDA
    yolov5.is_half = yolov5.device.type != 'cpu'
    if yolov5.is_half:
        yolov5.half()
    yolov5.imgsz = check_img_size(imgsz, s=yolov5.stride.max())
    yolov5.names = (yolov5.module.names if hasattr(
        yolov5, 'module') else yolov5.names)

    yolov5.augment = False
    yolov5.conf_thres = conf_thres
    yolov5.iou_thres = iou_thres
    yolov5.classes = None if track_only == [] else track_only
    yolov5.agnostic_nms = False

    return yolov5


def xyxy2tlwh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else zeros_like(x)
    y[:, 0] = x[:, 0]  # top left x
    y[:, 1] = x[:, 1]  # top left y
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def plot_one_box_tlwh(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[0]) + int(x[2]), int(x[1]) + int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def yolo_predict(yolov5, img, im0s,log):
    img = torch.from_numpy(img).to(yolov5.device)
    img = img.half() if yolov5.is_half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    t1 = time_synchronized()
    pred = yolov5(img, augment=yolov5.augment)[0]

    # Apply NMS
    pred = non_max_suppression(
        pred, yolov5.conf_thres, yolov5.iou_thres, classes=yolov5.classes, agnostic=yolov5.agnostic_nms)
    t2 = time_synchronized()

    boxes, class_inds, scores = [], [], []

    # Process detections
    for _, det in enumerate(pred):  # detections per image
        s, im0 = '', im0s

        s += '%gx%g ' % img.shape[2:]  # print string
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                s += '%g %ss, ' % (n, yolov5.names[int(c)])  # add to string

            # Write results
            for *xyxy, conf, cls in reversed(det):
                tlwh = (xyxy2tlwh(torch.tensor(xyxy).view(1, 4))
                        ).view(-1).tolist()  # xywh

                boxes.append(tlwh)
                class_inds.append(int(cls.item()))
                scores.append(conf.item())

        # Print time (inference + NMS)
        log.info('%sDone. (%.3fs)' % (s, t2 - t1))

    return boxes, class_inds, scores


def detect(save_img=False):
    out, source, imgsz, device = opt.save_dir, opt.source, opt.img_size, opt.device
    webcam = source.isnumeric() or source.startswith(
        ('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

    # Initialize
    set_logging()
    if os.path.exists(out):  # output dir
        shutil.rmtree(out)  # delete dir
    os.makedirs(out)  # make new dir

    # Load model
    desired_classes = [0, 1, 2, 3, 5, 7]
    yolov5 = Load_Yolo_Model(device=select_device(device), track_only=desired_classes)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in yolov5.names]

    # Set Dataloader
    yolov5.vid_path, yolov5.vid_writer = None, None
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        yolov5.dataset = LoadStreams(source, img_size=imgsz)
    else:
        yolov5.dataset = LoadImages(source, img_size=imgsz)

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=yolov5.device)  # init img
    _ = yolov5(img.half() if yolov5.is_half else img) if yolov5.is_half else None

    for path, img, im0s, _ in yolov5.dataset:
        p = Path(path)
        save_path = str(out + '/' + p.name)
        # path is the path of the image file, img is the formatted image, im0s is the original image from cv2.imread(path) in BGR format
        boxes, classes, scores = yolo_predict(yolov5, img, im0s)

        for i in range(len(boxes)):
            plot_one_box_tlwh(boxes[i], im0s, label=(yolov5.names[classes[i]] + " %.2f" % scores[i]), color=colors[int(classes[i])], line_thickness=3)

        cv2.imwrite(save_path, im0s)


    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='models/yolov5s.pt', help='model.pt path(s)')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str,
                        default='inference/output', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    opt = parser.parse_args()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
