# Face Recognition with Ros

- This repo illustrates how to use face-recogntion module with usb_cam by Ros
- We need Ros, pytorch, Cvbridge, Etc.
- requirments must be installed (see Reference).

## Ros
- We will skip step of basic Ros settigs.
- Architecture
    - Publish usb_cam images with Ros
        ```
        roslaunch usb_cam usb_cam-test.launch
        ```
    - Subscribe image topics in code, and convert it to Opencv_images with CVbridge
    - with Opencv images do face recognition
    - Publish result class, score, landmarks 

## How to use (you must launch 'usb_cam')
### 1. Upload Face you want to recognize in Face Bank
- #### take a picture
    ```
    rosrun new_face_recognition take_picture.py -n name
    ```
    - Enter 't' to take a picture
    - Enter 'q' to close window
    - Write name behind '-n' option to create file with name
-  #### upload picture file
    ```
    python Take_id.py --image {file path} --name {name} 
    ```
    - {file path} is path to picture file
    - {name} is name you want to use in face recognition
### 2. Save Faces in Face bank to model
```
rosrun new_face_recognition facebank.py
```
### 3. Run Face Recognition
#### - Run module
    rosrun new_face_recognition face_recognition.py

#### - Publish msg to activate 
    rostopic pub --once /face_recognition_msg std_msgs/String "data: 'On'"




## Reference
- Face Recognition(MTCNN) link: https://github.com/xuexingyu24/MobileFaceNet_Tutorial_Pytorch
     ```
    git clone https://github.com/xuexingyu24/MobileFaceNet_Tutorial_Pytorch
    ```
- Usb_cam link: https://github.com/ros-drivers/usb_cam.git
    ```
    git clone https://github.com/ros-drivers/usb_cam.git
    ```
