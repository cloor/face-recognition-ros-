#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class Image_generator:
    def __init__(self):
        self.sub_usb_image = rospy.Subscriber('/usb_cam/image_raw', Image, self.callback)
        self.pub_camera_image = rospy.Publisher('/camera/color/image_raw', Image, queue_size = 5)
        self.bridge = CvBridge()

    def callback(self, image):

        self.pub_camera_image.publish(image)

if __name__== '__main__':
    print('start')
    rospy.init_node("iamge_generator_node", anonymous=True)
    t = Image_generator()
    rospy.spin()