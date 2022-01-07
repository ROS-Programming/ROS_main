#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import cv2
import numpy as np
import mediapipe as mp
import glob
import create_user_dataset
import os
from os import listdir
from os.path import isfile, join

def create_model(data_path):  
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    onlyfiles = onlyfiles[1:]
    Training_Data, Labels = [], []
    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    Labels = np.asarray(Labels, dtype=np.int32)
    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    return model

def face_detector(img):
    ret = 0
    face_img = np.zeros(IMG_SIZE, dtype=np.uint8)
    with faceModule.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        shape = img.shape

        if result.multi_face_landmarks:
            for res in result.multi_face_landmarks:
                rectangle_data_l = [int(res.landmark[123].x * shape[1]), int(res.landmark[352].x * shape[1]),
                                    int(res.landmark[10].y * shape[0]), int(res.landmark[152].y * shape[0])]
                cv2.rectangle(img, (rectangle_data_l[0]-10, rectangle_data_l[2] - 10), (rectangle_data_l[1] + 10, rectangle_data_l[3] + 10 ), color=(255, 0, 0), thickness=2)

                face_img = img[rectangle_data_l[2]:rectangle_data_l[3], rectangle_data_l[0]:rectangle_data_l[1]]
                face_img = cv2.resize(face_img, dsize=IMG_SIZE)
                ret = 1
    return face_img, ret

def talker():
    data_path = '/home/pi/Desktop/ROS/ros_main/src/BMO/src/faces/'
    image_path = glob.glob("/home/pi/Desktop/ROS/ros_main/src/BMO/src/faces/*.png")
    if len(image_path) == 0:
        create_user_dataset.create_dataset()
    model = create_model(data_path)
    print("Model Training Complete!!!!!")

    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    IMG_SIZE = (200, 200)
    face_check = 0
    cap = cv2.VideoCapture(0)
    pub = rospy.Publisher('AI', String, queue_size=10)
    rospy.init_node('AI', anonymous=True)
    rate = rospy.Rate(10)
    while (not rospy.is_shutdown()):
        ret, frame = cap.read()

        face, check = face_detector(frame)
        if check == 1:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)

            if result[1] < 500:
                confidence = int(100*(1-(result[1])/300))
                display_string = str(confidence)+'% Confidence it is user'
            cv2.putText(frame,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)

            if confidence > 75:
                cv2.putText(frame, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', frame)
                face_check = 1
            else:
                cv2.putText(frame, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Cropper', frame)
                face_check = 0
        else:
            cv2.putText(frame, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Cropper', frame)
            face_check = 0
        AI_str = "AI %d" % face_check
        pub.publish(AI_str)
        rate.sleep()
        check = cv2.waitKey(1)

        if check == ord('r'):
            cap.release()
            cv2.destroyAllWindows()
            image_path = glob.glob("/home/pi/Desktop/ROS/ros_main/src/BMO/src/faces/*.png")
            for i in image_path:
                os.remove(i)
            create_user_dataset.create_dataset()
            data_path = '/home/pi/Desktop/ROS/ros_main/src/BMO/src/faces/'
            model = create_model(data_path)
            cap = cv2.VideoCapture(0)

        if check == ord('a'):
            cap.release()
            cv2.destroyAllWindows()
            image_path = glob.glob("/home/pi/Desktop/ROS/ros_main/src/BMO/src/faces/*.png")
            create_user_dataset.create_dataset(len(image_path))
            data_path = '/home/pi/Desktop/ROS/ros_main/src/BMO/src/faces/'
            model = create_model(data_path)
            cap = cv2.VideoCapture(0)

        if check == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

IMG_SIZE = (200, 200)
faceModule = mp.solutions.face_mesh

try:
    talker()
except rospy.ROSInterruptException:
    pass