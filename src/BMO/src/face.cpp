#include "ros/ros.h"
#include "std_msgs/String.h"

void chatterCallback(const std_msgs::String::ConstPtr& msg){
    ROS_INFO("face: [%s]", msg->data.c_str());
}

int main(int argc, char **argv){
    ros::init(argc, argv, "face");
    ros::NodeHandle n;
    ros::Subscriber sub_button = n.subscribe("button", 1000, chatterCallback);
    ros::Subscriber sub_AI = n.subscribe("AI", 1000, chatterCallback);
    ros::spin();
    return 0;
}