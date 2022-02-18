from __future__ import division
import os
import cv2
import dlib
from .eye import Eye
from .calibration import Calibration
import time
import onnx
import vision.utils.box_utils_numpy as box_utils
from caffe2.python.onnx import backend
import numpy as np


import onnxruntime as ort
# import libraries for landmark
from common.utils import BBox
from PIL import Image
import torchvision.transforms as transforms
import onnxruntime

# setup the parameters
resize = transforms.Resize([56, 56])
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
# import the landmark detection models
onnx_model_landmark = onnx.load("onnx/landmark_detection_56_se_external.onnx")
onnx.checker.check_model(onnx_model_landmark)
ort_session_landmark = onnxruntime.InferenceSession("onnx/landmark_detection_56_se_external.onnx")
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


threshold = 0.7

# face detection setting
def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = box_utils.hard_nms(box_probs,
                                       iou_threshold=iou_threshold,
                                       top_k=top_k,
                                       )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

label_path = "models/voc-model-labels.txt"

onnx_path = "models/onnx/version-RFB-320.onnx"
class_names = [name.strip() for name in open(label_path).readlines()]

predictor = onnx.load(onnx_path)
onnx.checker.check_model(predictor)
onnx.helper.printable_graph(predictor.graph)
predictor = backend.prepare(predictor, device="CPU")  # default CPU

ort_session = ort.InferenceSession(onnx_path)
input_name = ort_session.get_inputs()[0].name

class GazeTracking(object):
    """
    This class tracks the user's gaze.
    It provides useful information like the position of the eyes
    and pupils and allows to know if the eyes are open or closed
    """

    def __init__(self):
        self.frame = None
        self.eye_left = None
        self.eye_right = None
        self.calibration = Calibration()
    ### 68 landmarks detection
    def _analyze(self):
        image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (320, 240))
        # image = cv2.resize(image, (640, 480))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = image.astype(np.float32)
        # confidences, boxes = predictor.run(image)
        # time_time = time.time()
        confidences, boxes = ort_session.run(None, {input_name: image})
        # print("cost time:{}".format(time.time() - time_time))
        boxes, labels, probs = predict(self.frame.shape[1], self.frame.shape[0], confidences, boxes, threshold)   
        for i in range(boxes.shape[0]):
            box = boxes[i, :]
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"

            #cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            # perform landmark detection
            out_size = 224
            img=self.frame.copy()
            height,width,_=img.shape
            x1=box[0]
            y1=box[1]
            x2=box[2]
            y2=box[3]
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            size = int(max([w, h])*1.1)
            cx = x1 + w//2
            cy = y1 + h//2
            x1 = cx - size//2
            x2 = x1 + size
            y1 = cy - size//2
            y2 = y1 + size
            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)   
            new_bbox = list(map(int, [x1, x2, y1, y2]))
            new_bbox = BBox(new_bbox)
            cropped=img[new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]
            if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)            
            cropped_face = cv2.resize(cropped, (out_size, out_size))

            if cropped_face.shape[0]<=0 or cropped_face.shape[1]<=0:
                continue
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)    
            cropped_face = Image.fromarray(cropped_face)
            test_face = resize(cropped_face)
            test_face = to_tensor(test_face)
            test_face = normalize(test_face)
            test_face.unsqueeze_(0)

            # start = time.time()             
            ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(test_face)}
            ort_outs = ort_session_landmark.run(None, ort_inputs)
            # end = time.time()
            # print('Time: {:.6f}s.'.format(end - start))
            landmark = ort_outs[0]
            landmark = landmark.reshape(-1,2)
            landmarks = new_bbox.reprojectLandmark(landmark)

            try:
                self.eye_left = Eye(self.frame, landmarks, 0, self.calibration)
                self.eye_right = Eye(self.frame, landmarks, 1, self.calibration)
            
            except IndexError:
                self.eye_left = None
                self.eye_right = None
    @property
    def pupils_located(self):
        """Check that the pupils have been located"""
        try:
            int(self.eye_left.pupil.x)
            int(self.eye_left.pupil.y)
            int(self.eye_right.pupil.x)
            int(self.eye_right.pupil.y)
            return True
        except Exception:
            return False

    #def _analyze(self):
    #    """Detects the face and initialize Eye objects"""
    #    frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
    #    faces = self._face_detector(frame)
    #    
    #    try:
    #        landmarks = self._predictor(frame, faces[0])
    #        
    #        self.eye_left = Eye(frame, landmarks, 0, self.calibration)
    #        self.eye_right = Eye(frame, landmarks, 1, self.calibration)
    #        
    #    except IndexError:
    #        self.eye_left = None
    #        self.eye_right = None

    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): The frame to analyze
        """
        self.frame = frame
        self._analyze()

    def pupil_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.pupils_located:
            x = self.eye_left.origin[0] + self.eye_left.pupil.x
            y = self.eye_left.origin[1] + self.eye_left.pupil.y
            return (x, y)

    def pupil_right_coords(self):
        """Returns the coordinates of the right pupil"""
        if self.pupils_located:
            x = self.eye_right.origin[0] + self.eye_right.pupil.x
            y = self.eye_right.origin[1] + self.eye_right.pupil.y
            return (x, y)

    def horizontal_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        horizontal direction of the gaze. The extreme right is 0.0,
        the center is 0.5 and the extreme left is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.x / (self.eye_left.center[0] * 2 - 10)
            pupil_right = self.eye_right.pupil.x / (self.eye_right.center[0] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def vertical_ratio(self):
        """Returns a number between 0.0 and 1.0 that indicates the
        vertical direction of the gaze. The extreme top is 0.0,
        the center is 0.5 and the extreme bottom is 1.0
        """
        if self.pupils_located:
            pupil_left = self.eye_left.pupil.y / (self.eye_left.center[1] * 2 - 10)
            pupil_right = self.eye_right.pupil.y / (self.eye_right.center[1] * 2 - 10)
            return (pupil_left + pupil_right) / 2

    def is_right(self):
        """Returns true if the user is looking to the right"""
        if self.pupils_located:
            return self.horizontal_ratio() <= 0.3

    def is_left(self):
        """Returns true if the user is looking to the left"""
        if self.pupils_located:
            return self.horizontal_ratio() >= 0.75

    def is_center(self):
        """Returns true if the user is looking to the center"""
        if self.pupils_located:
            return self.is_right() is not True and self.is_left() is not True

    def is_blinking(self):
        """Returns true if the user closes his eyes"""
        if self.pupils_located:
            blinking_ratio = (self.eye_left.blinking + self.eye_right.blinking) / 2
            return blinking_ratio > 3.8

    def annotated_frame(self):
        """Returns the main frame with pupils highlighted"""
        frame = self.frame.copy()

        if self.pupils_located:
            color = (0, 255, 0)
            x_left, y_left = self.pupil_left_coords()
            x_right, y_right = self.pupil_right_coords()
            cv2.line(frame, (x_left - 5, y_left), (x_left + 5, y_left), color)
            cv2.line(frame, (x_left, y_left - 5), (x_left, y_left + 5), color)
            cv2.line(frame, (x_right - 5, y_right), (x_right + 5, y_right), color)
            cv2.line(frame, (x_right, y_right - 5), (x_right, y_right + 5), color)
            
        return frame
