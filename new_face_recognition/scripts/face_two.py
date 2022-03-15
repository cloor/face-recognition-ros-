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
from utils.align_trans import *
from MTCNN import create_mtcnn_net
from face_model import MobileFaceNet
from facebank import load_facebank
import cv2
import time

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as Image_msg
from vision_msg.msg import face_recognition_result
from std_msgs.msg import String



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

def ensure_color(image):
    if len(image.shape) == 2:
        return np.dstack([image] * 3)
    elif image.shape[2] == 1:
        return np.dstack([image] * 3)
    return image

configs = json.load(open("catkin_ws/src/new_face_recognition/scripts/configs/fer2013_config.json"))
image_size=(configs["image_size"], configs["image_size"])
model = densenet121(in_channels=3, num_classes=7)
model.cpu()
state = torch.load('catkin_ws/src/new_face_recognition/scripts/saved/checkpoints/densenet121_test_2022Feb11_17.38')
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

    bboxes, landmarks = [], []
    try:
        bboxes, landmarks = create_mtcnn_net(input, args.mini_face, device, 
                                            p_model_path='catkin_ws/src/new_face_recognition/scripts/MTCNN/weights/pnet_Weights',
                                            r_model_path='catkin_ws/src/new_face_recognition/scripts/MTCNN/weights/rnet_Weights',
                                            o_model_path='catkin_ws/src/new_face_recognition/scripts/MTCNN/weights/onet_Weights')
    except:
        pass
    # FPS = 1.0 / (time.time() - start_time)
    if len(bboxes) != 0:
        bboxes = bboxes / args.scale
        landmarks = landmarks / args.scale
    
    faces = Face_alignment(frame_bgr, default_square=True, landmarks=landmarks) 
    
    embs = []

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



        # Put Circle on landmarks
        for p in landmarks:
            for i in range(5):
                frame_bgr = cv2.circle(frame_bgr, (int(p[i]), int(p[i + 5])),1, (255,0,0), 2)
    
    # Calculate FPS
    FPS = 1.0 / (time.time() - start_time)
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