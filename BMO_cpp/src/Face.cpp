#include <functional>
#include <memory>
#include <string>
#include <iostream>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "opencv2/opencv.hpp"
#include <algorithm>

using namespace cv;
using namespace std;
using std::placeholders::_1;

class Face_main : public rclcpp::Node{
	public:
		Face_main() : Node("Face_main"){
			auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(10));
			Face_sub_ = this->create_subscription<std_msgs::msg::String>(
				"AI",
				qos_profile,
				std::bind(&Face_main::subscribe_topic_message, this, _1));
		}
	private:
		void subscribe_topic_message(const std_msgs::msg::String::SharedPtr msg) const{
			RCLCPP_INFO(this->get_logger(), "Data: '%s'", msg->data.c_str());
			String msg_data = msg->data;
			Mat img_color;
			if (msg_data == "AI 1"){
				printf("unlocked\n");
				img_color = imread("/home/ubuntu/Desktop/ROS/ros_main/BMO_cpp/src/BMO_Faces/1.jpg", IMREAD_COLOR);
				imshow("BMO", img_color);
			}
			else if (msg_data == "AI -1"){
				printf("face not found\n");
				img_color = imread("/home/ubuntu/Desktop/ROS/ros_main/BMO_cpp/src/BMO_Faces/3.jpg", IMREAD_COLOR);
				imshow("BMO", img_color);
			}
			else{
				printf("locked\n");
				img_color = imread("/home/ubuntu/Desktop/ROS/ros_main/BMO_cpp/src/BMO_Faces/5.jpg", IMREAD_COLOR);
				imshow("BMO", img_color);
			}
			if (waitKey(1) == 27){
				printf("esc input\n");
			}
		}
		rclcpp::Subscription<std_msgs::msg::String>::SharedPtr Face_sub_;

};

int main(int argc, char * argv[])
{
  namedWindow("BMO");
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Face_main>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  destroyAllWindows();
  return 0;
}
