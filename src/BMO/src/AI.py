#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import cv2

def talker():
    cap = cv2.VideoCapture(0)
    pub = rospy.Publisher('AI', String, queue_size=10)
    rospy.init_node('AI', anonymous=True)
    rate = rospy.Rate(10)
    turn = 0 
    while (not rospy.is_shutdown()):
        ret, frame = cap.read()
        AI_str = "AI %d" % turn
        turn += 1
        rospy.loginfo(AI_str)
        pub.publish(AI_str)
        cv2.imshow("webcam", frame)
        rate.sleep()
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass