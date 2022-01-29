#include <stdio.h>
#include <string.h>
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

int main(int argc, char * argv[])
{
	rclcpp::init(argc, argv);
  FILE *file;
  //Opening device file

  int getnum;

  while (true)
    {
      file = fopen("/dev/ttyACM1 (Arduino Micro)", "w");
      cout << ">>" << endl;
      cin >> getnum;
      fprintf(file, "%d", getnum); //Writing to the file
      fclose(file);
    }

}