#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: AIRocker
"""

import sys
import os

import json
from models import densenet121


sys.path.append(os.path.join(sys.path[0], 'MTCNN'))
import argparse
import torch
from torchvision import transforms as trans
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from util.align_trans import *
from MTCNN import create_mtcnn_net
from face_model import MobileFaceNet
from facebank import load_facebank
import cv2
import time


from typing import List, Optional
from models.eyenet import EyeNet
import dlib
import imutils
import util.gaze
from imutils import face_utils
from util.eye_prediction import EyePrediction
from util.eye_sample import EyeSample



import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as Image_msg
from vision_msg.msg import face_recognition_result
from std_msgs.msg import String


device = torch.device("cpu")
# EMOTION
transform = trans.Compose([trans.ToPILImage(), trans.ToTensor()])

FER_2013_EMO_DICT = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}





# gaze
dirname = os.path.dirname(__file__)
landmarks_detector = dlib.shape_predictor(os.path.join(dirname, 'Weights/shape_predictor_5_face_landmarks.dat'))
checkpoint = torch.load('catkin_ws/src/new_face_recognition/scripts/Weights/checkpoint.pt', map_location=device)
nstack = checkpoint['nstack']
nfeatures = checkpoint['nfeatures']
nlandmarks = checkpoint['nlandmarks']
eyenet = EyeNet(nstack=nstack, nfeatures=nfeatures, nlandmarks=nlandmarks).to(device)
eyenet.load_state_dict(checkpoint['model_state_dict'] , strict = False)


def detect_landmarks(b, frame, scale_x=0, scale_y=0):
    rectangle = dlib.rectangle(int(b[0]), int(b[1]), int(b[2]), int(b[3]))
    
    face_landmarks = landmarks_detector(frame, rectangle)
    
    return face_utils.shape_to_np(face_landmarks)

def segment_eyes(frame, landmarks, ow=160, oh=96):
    eyes = []

    # Segment eyes
    for corner1, corner2, is_left in [(2, 3, True), (0, 1, False)]:
        x1, y1 = landmarks[corner1, :]
        x2, y2 = landmarks[corner2, :]
        eye_width = 1.5 * np.linalg.norm(landmarks[corner1, :] - landmarks[corner2, :])
        
        if eye_width == 0.0:
            return eyes

        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

        # center image on middle of eye
        translate_mat = np.asmatrix(np.eye(3))
        translate_mat[:2, 2] = [[-cx], [-cy]]
        inv_translate_mat = np.asmatrix(np.eye(3))
        inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

        # Scale
        scale = ow / eye_width
        scale_mat = np.asmatrix(np.eye(3))
        scale_mat[0, 0] = scale_mat[1, 1] = scale
        inv_scale = 1.0 / scale
        inv_scale_mat = np.asmatrix(np.eye(3))
        inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale

        estimated_radius = 0.5 * eye_width * scale

        # center image
        center_mat = np.asmatrix(np.eye(3))
        center_mat[:2, 2] = [[0.5 * ow], [0.5 * oh]]
        inv_center_mat = np.asmatrix(np.eye(3))
        inv_center_mat[:2, 2] = -center_mat[:2, 2]

        # Get rotated and scaled, and segmented image
        transform_mat = center_mat * scale_mat * translate_mat
        inv_transform_mat = (inv_translate_mat * inv_scale_mat * inv_center_mat)

        eye_image = cv2.warpAffine(frame, transform_mat[:2, :], (ow, oh))
        eye_image = cv2.equalizeHist(eye_image)
        if is_left:
            eye_image = np.fliplr(eye_image)
            cv2.imshow('left eye image', eye_image)
        else:
            cv2.imshow('right eye image', eye_image)
        
        eyes.append(EyeSample(orig_img=frame.copy(),
                              img=eye_image,
                              transform_inv=inv_transform_mat,
                              is_left=is_left,
                              estimated_radius=estimated_radius))
    return eyes



def gaze(horizon,vertical):
    if horizon<=0.33:
        x = 'out of center'
    elif horizon>= 0.66:
        x = 'out of center'
    elif vertical<= 0.3:
        x = 'out of center'
    elif vertical>=0.7:
        x = 'out of center'
    else:
        x = 'center'
    return x



def run_eyenet(eyes: List[EyeSample], ow=160, oh=96) -> List[EyePrediction]:
    result = []
    for eye in eyes:
        with torch.no_grad():
            x = torch.tensor([eye.img], dtype=torch.float32).to(device)
            _, landmarks= eyenet.forward(x)
            landmarks = np.asarray(landmarks.cpu().numpy()[0])
            # gaze = np.asarray(gaze.cpu().numpy()[0])
            # assert gaze.shape == (2,)
            assert landmarks.shape == (34, 2)

            landmarks = landmarks * np.array([oh/48, ow/80])

            temp = np.zeros((34, 3))
            if eye.is_left:
                temp[:, 0] = ow - landmarks[:, 1]
            else:
                temp[:, 0] = landmarks[:, 1]
            temp[:, 1] = landmarks[:, 0]
            temp[:, 2] = 1.0
            landmarks = temp
            assert landmarks.shape == (34, 3)
            landmarks = np.asarray(np.matmul(landmarks, eye.transform_inv.T))[:, :2]
            assert landmarks.shape == (34, 2)
            result.append(EyePrediction(eye_sample=eye, landmarks=landmarks)) # gaze=gaze))
    return result

def smooth_eye_landmarks(eye: EyePrediction, prev_eye: Optional[EyePrediction], smoothing=0.2): # , gaze_smoothing=0.4):
    if prev_eye is None:
        return eye
    return EyePrediction(
        eye_sample=eye.eye_sample,
        landmarks=smoothing * prev_eye.landmarks + (1 - smoothing) * eye.landmarks)
        # gaze=gaze_smoothing * prev_eye.gaze + (1 - gaze_smoothing) * eye.gaze)




def ensure_color(image):
    if len(image.shape) == 2:
        return np.dstack([image] * 3)
    elif image.shape[2] == 1:
        return np.dstack([image] * 3)
    return image

configs = json.load(open("catkin_ws/src/new_face_recognition/scripts/Weights/fer2013_config.json"))
image_size=(configs["image_size"], configs["image_size"])
model = densenet121(in_channels=3, num_classes=7)
model.cpu()
state = torch.load('catkin_ws/src/new_face_recognition/scripts/Weights/densenet121_test_2022Mar15_17.44')
model.load_state_dict(state["net"])
model.eval()


# CV bridge : OpenCV 와 ROS 를 이어주는 역할 
bridge = CvBridge()

# initialize result publisher
result_pub = rospy.Publisher('face_recognition_result', face_recognition_result, queue_size=10)

def resize_image(img, scale):
    """
        resize image
    """
    height, width, channel = img.shape
    new_height = int(height * scale)     # resized new height
    new_width = int(width * scale)       # resized new width
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)      # resized image
    return img_resized


def message_callback(message):
    global flag
    if message == String("On"):
        flag = 1
    else:
        flag = 0

        


def camera_callback():

    # Get ROS image using wait for message
    image = rospy.wait_for_message('/camera/color/image_raw', Image_msg)
    frame_bgr = bridge.imgmsg_to_cv2(image, desired_encoding = 'bgr8')
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    start_time = time.time()
    input = resize_image(frame_bgr, args.scale)
    alpha = 0.95
    landmarks_gaze = None
    left_eye = None
    right_eye = None
    bboxes, landmarks = [], []
    try:
        bboxes, landmarks = create_mtcnn_net(input, args.mini_face, device, 
                                            p_model_path='catkin_ws/src/new_face_recognition/scripts/Weights/pnet_Weights',
                                            r_model_path='catkin_ws/src/new_face_recognition/scripts/Weights/rnet_Weights',
                                            o_model_path='catkin_ws/src/new_face_recognition/scripts/Weights/onet_Weights')
    except:
        pass
    # FPS = 1.0 / (time.time() - start_time)
    if len(bboxes) != 0:
        bboxes = bboxes / args.scale
        landmarks = landmarks / args.scale
    
    faces = Face_alignment(frame_bgr, default_square=True, landmarks=landmarks) 
    
    embs = []
    Eye = None
    for img in faces:  
        embs.append(detect_model(test_transform(img).to(device).unsqueeze(0)))
        
    if len(embs) >= 1:
        source_embs = torch.cat(embs)  # number of detected faces x 512
        diff = source_embs.unsqueeze(-1) - targets.transpose(1, 0).unsqueeze(0) # i.e. 3 x 512 x 1 - 1 x 512 x 2 = 3 x 512 x 2
        dist = torch.sum(torch.pow(diff, 2), dim=1) # number of detected faces x numer of target faces
        minimum, min_idx = torch.min(dist, dim=1) # min and idx for each row
        min_idx[minimum > ((args.threshold-156)/(-80))] = -1  # if no match, set idx to -1
        score = minimum
        results = min_idx
        
        names[0] = 'unknown'

        # convert distance to score dis(0.7,1.2) to score(100,60)
        score_100 = torch.clamp(score*-80+156,0,100)

        for i, b in enumerate(bboxes):
            # Put Face Bounding box 
            
            frame_bgr = cv2.rectangle(frame_bgr, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255,0,0), 1)
            # Put Text about name and score
            frame_bgr = cv2.putText(frame_bgr, names[results[i] + 1]+" / Score : {:.0f}".format(score_100[i]),
                    (int(b[0]), int(b[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255)) 
            
            
            # Emotion
            face = gray[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
            
            face = ensure_color(face)

            face = cv2.resize(face, image_size)
            face = transform(face).cpu()
                   
            face = torch.unsqueeze(face, dim=0) # size(1,3,224,224)
                    
            output = torch.squeeze(model(face), 0)
                    
            proba = torch.softmax(output, 0)
            emo_proba, emo_idx = torch.max(proba, dim=0)       
            emo_idx = emo_idx.item()
            emo_proba = emo_proba.item()
            emo_label = FER_2013_EMO_DICT[emo_idx]
            cv2.putText(
                        frame_bgr,
                        "{} {}".format(emo_label, int(emo_proba * 100)),
                        (int(b[2]), int(b[1]) + 1),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 0),
                        2,
                    )

            # Gaze
            next_landmarks = detect_landmarks(b, gray)

            if landmarks_gaze is not None:
                landmarks_gaze = next_landmarks * alpha + (1 - alpha) * landmarks_gaze
            else:
                landmarks_gaze = next_landmarks

            if landmarks_gaze is not None:
                eye_samples = segment_eyes(gray, landmarks_gaze)

                eye_preds = run_eyenet(eye_samples)
                left_eyes = list(filter(lambda x: x.eye_sample.is_left, eye_preds))
                right_eyes = list(filter(lambda x: not x.eye_sample.is_left, eye_preds))

                if left_eyes:
                    left_eye = smooth_eye_landmarks(left_eyes[0], left_eye, smoothing=0.1)
                if right_eyes:
                    right_eye = smooth_eye_landmarks(right_eyes[0], right_eye, smoothing=0.1)

                ratio=[]
                ratio_2=[]
                for ep in [left_eye, right_eye]:
                    landmarks_array_x = []
                    landmarks_array_y = []
                    
                    for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,32]:
                        for (x, y) in ep.landmarks[i:i+1]:
                            landmarks_array_x.append(x)
                            landmarks_array_y.append(y)
                            if i == 32:
                                color = (0, 255, 0)
                                if ep.eye_sample.is_left:
                                    color = (255, 0, 0)
                                cv2.circle(frame_bgr,
                                        (int(round(x)), int(round(y))), 1, color, -1, lineType=cv2.LINE_AA)
                    width=abs(min(landmarks_array_x)-max(landmarks_array_x))
                    height=abs(min(landmarks_array_y)-max(landmarks_array_y))
                    center=(landmarks_array_x[-1],landmarks_array_y[-1])
                    center_x=abs(center[0]-min(landmarks_array_x))
                    center_y=abs(center[1]-min(landmarks_array_y))
                    horizontal = center_x / width
                    vertical = center_y / height
                    ratio.append(horizontal)
                    ratio_2.append(vertical)
                Eye = gaze(horizon=np.mean(ratio),vertical=np.mean(ratio_2))




                # for ep in [left_eye, right_eye]:
                #     for (x, y) in ep.landmarks[16:33]:
                #         color = (0, 255, 0)
                #         if ep.eye_sample.is_left:
                #             color = (255, 0, 0)
                #         cv2.circle(frame_bgr,
                #                 (int(round(x)), int(round(y))), 1, color, -1, lineType=cv2.LINE_AA)

#                    gaze = ep.gaze.copy()
#                    if ep.eye_sample.is_left:
#                        gaze[1] = -gaze[1]
#                    util.gaze.draw_gaze(frame_bgr, ep.landmarks[-2], gaze, length=60.0, thickness=2)

        # Put Circle on landmarks
        # for p in landmarks:
        #     for i in range(5):
        #         frame_bgr = cv2.circle(frame_bgr, (int(p[i]), int(p[i + 5])),1, (255,0,0), 2)
    
    # Calculate FPS
    FPS = 1.0 / (time.time() - start_time)
    frame_bgr = cv2.putText(frame_bgr, Eye, (90,60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    frame_bgr = cv2.putText(frame_bgr,'FPS: {:.1f}'.format(FPS),(10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0))
            
    cv2.imshow('video', frame_bgr)

    cv2.waitKey(1)
    

    # Topic Publish
    try:
        fr_result = face_recognition_result()
        
        fr_result.num_face = str(len(bboxes))
        
        for num, res in enumerate(bboxes):
            fr_result.names += names[results[num] + 1] +' '
            fr_result.face_xmin += str(int(res[0])) + ' '
            fr_result.face_ymin += str(int(res[1])) + ' '
            fr_result.face_xmax += str(int(res[2])) + ' '
            fr_result.face_ymax += str(int(res[3])) + ' '
            fr_result.face_score += str(int(score_100[num].item())) + ' '

        result_pub.publish(fr_result)
    
    except:
        pass

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='face detection demo')
    parser.add_argument('-th','--threshold',help='threshold score to decide identical faces',default=60, type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank",action="store_true", default= False)
    parser.add_argument("-tta", "--tta", help="whether test time augmentation",action="store_true", default= False)
    parser.add_argument("-c", "--score", help="whether show the confidence score",action="store_true",default= True )
    parser.add_argument("--scale", dest='scale', help="input frame scale to accurate the speed", default=0.5, type=float)
    parser.add_argument('--mini_face', dest='mini_face', help=
    "Minimum face to be detected. derease to increase accuracy. Increase to increase speed",
                        default=40, type=int)
    args = parser.parse_args()

    # set device using only cpu
    device = torch.device("cpu")

    # set detect model
    detect_model = MobileFaceNet(512).to(device)  # embeding size is 512 (feature vector)
    detect_model.load_state_dict(torch.load('catkin_ws/src/new_face_recognition/scripts/Weights/MobileFace_Net', map_location=lambda storage, loc: storage))
    print('MobileFaceNet face detection model generated')
    detect_model.eval()

    # Load Facebank data
    targets, names = load_facebank(path='catkin_ws/src/new_face_recognition/scripts/facebank')
    print('facebank loaded')
    
    # Set transform 
    test_transform = trans.Compose([
                        trans.ToTensor(),
                        trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
   
    # ros node 
    rospy.init_node('face_recognition', anonymous = True)
    

    global flag 
    flag = 0

    while True:
        message_sub = rospy.Subscriber('/face_recognition_msg', String, message_callback)
        if flag == 1:
            camera_callback()
    cv2.destroyAllWindows()
