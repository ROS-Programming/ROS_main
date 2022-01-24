#include <chrono>
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
using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
	rclcpp::init(argc, argv);
	Mat img_color;
	int val = 1;
	img_color = imread("/home/ubuntu/Desktop/ROS/ros_main/BMO_cpp/src/BMO_Faces/" + to_string(val) +".jpg", IMREAD_COLOR);
	if (img_color.empty()) return -1;
	namedWindow("BMO");
	imshow("BMO", img_color);

	while (1)
	{
		int key = waitKey(1);
		if (key == 27) break;
		else if (key == 'a')
		{
			if (val <= 1) continue;
			else
			{
				val -= 1;
				img_color = imread("/home/ubuntu/Desktop/ROS/ros_main/BMO_cpp/src/BMO_Faces/" + to_string(val) + ".jpg", IMREAD_COLOR);
				if (img_color.empty()) return -1;
				namedWindow("BMO");
				imshow("BMO", img_color);
			}
		}
		else if (key == 'd')
		{
			if (val >= 9) continue;
			else
			{
				val += 1;
				img_color = imread("/home/ubuntu/Desktop/ROS/ros_main/BMO_cpp/src/BMO_Faces/" + to_string(val) + ".jpg", IMREAD_COLOR);
				if (img_color.empty()) return -1;
				namedWindow("BMO");
				imshow("BMO", img_color);
			}
		}
	}

	destroyAllWindows();
	return 0;
}