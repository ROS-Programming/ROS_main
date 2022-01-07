#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import mediapipe as mp 

def talker():
    pub = rospy.Publisher('test', String, queue_size=10)
    rospy.init_node('test', anonymous=True)
    rate = rospy.Rate(10)
    turn = 0 
    while (not rospy.is_shutdown()):
        AI_str = "test %d" % turn
        turn += 1
        rospy.loginfo(AI_str)
        pub.publish(AI_str)
        rate.sleep()
if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass