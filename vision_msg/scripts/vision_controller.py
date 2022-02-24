#!/usr/bin/env python

from scipy.optimize.optimize import bracket
import rospy

from vision_msg.msg import vision_msg
from std_msgs.msg import String

human_detection_pub = rospy.Publisher('human_detection_msg', String, queue_size = 10)
face_recognition_pub = rospy.Publisher('face_recognition_msg', String, queue_size = 10)
action_recognition_pub = rospy.Publisher('action_recognition_msg', String, queue_size = 10)

def call_back(message):
    
    while message.face_recognition_start == 'On':
        face_recognition_pub.publish(message.face_recognition_start)
        message = rospy.wait_for_message('/vision_msg', vision_msg)

    human_detection_pub.publish(message.human_detection_start)                  
    action_recognition_pub.publish(message.action_recognition_start)


    
def main():
    message = vision_msg()

    rospy.init_node("vision_msg_node", anonymous=True) 
    
    message.human_detection_start = 'Off'
    message.face_recognition_start = 'Off'
    message.action_recognition_start = 'Off'

    sub = rospy.Subscriber('/vision_msg', vision_msg, call_back)
    rospy.spin()

if __name__== '__main__':
    print('start')
    main()
    