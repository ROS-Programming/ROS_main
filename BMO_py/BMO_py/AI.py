import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String
import cv2
import numpy as np
import mediapipe as mp
import glob
import BMO_py.create_user_dataset as cud
import os
from os import listdir
from os.path import isfile, join

class AI_main(Node):
    def __init__(self):
        self.faceModule = mp.solutions.face_mesh
        data_path = '/home/ubuntu/Desktop/ROS/ros_main/BMO_py/BMO_py/faces/'
        image_path = glob.glob(data_path + "*.png")
        if len(image_path) == 0:
            cud.create_dataset()
        self.model = self.create_model(data_path)
        print("Model Training Complete!!!!!")

        self.face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.IMG_SIZE = (200, 200)

        super().__init__('AI_publisher')
        qos_profile = QoSProfile(depth=10)
        self.AI_publisher = self.create_publisher(String, 'AI', qos_profile)
        self.timer = self.create_timer(1, self.talker)

    def create_model(self, data_path):
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

    def face_detector(self, img):
        ret = 0
        face_img = np.zeros(self.IMG_SIZE, dtype=np.uint8)
        with self.faceModule.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            shape = img.shape
            try:
                if result.multi_face_landmarks:
                    for res in result.multi_face_landmarks:
                        rectangle_data_l = [int(res.landmark[123].x * shape[1]), int(res.landmark[352].x * shape[1]),
                                            int(res.landmark[10].y * shape[0]), int(res.landmark[152].y * shape[0])]
                        cv2.rectangle(img, (rectangle_data_l[0]-10, rectangle_data_l[2] - 10), (rectangle_data_l[1] + 10, rectangle_data_l[3] + 10 ), color=(255, 0, 0), thickness=2)

                        face_img = img[rectangle_data_l[2]:rectangle_data_l[3], rectangle_data_l[0]:rectangle_data_l[1]]
                        face_img = cv2.resize(face_img, dsize=self.IMG_SIZE)
                        ret = 1
                else:
                    ret = -1
            except:
                ret = -1
        return face_img, ret

    def talker(self):
        msg = String()
        face_check = 0
        cap = cv2.VideoCapture(0)
        while (True):
            ret, frame = cap.read()
            if ret == False:
                print("Please Connect Camera")
                continue
            face, check = self.face_detector(frame)
            if check == 1:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                result = self.model.predict(face)

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
            if (check == -1):
                msg.data = "AI %d" % check
            else:
                msg.data = "AI %d" % face_check
            self.AI_publisher.publish(msg)
            check = cv2.waitKey(1)

            if check == 27:
                break
        cap.release()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = AI_main()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard Interrupt (SIGINT)')
    finally:
        node.destroy_node()
        rclpy.shutdown()
    cv2.destroyAllWindows()

