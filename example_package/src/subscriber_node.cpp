#include<ros/ros.h>
#include<std_msgs/String.h>

void writeMsgToLog(const std_msgs::String::ConstPtr& msg) {
	ROS_INFO("The message received was: %s", msg->data.c_str());
}

int main(int argc, char** argv) {
	ros::init(argc, argv, "subscriber");
	ros::NodeHandle nh;

	ros::Subscriber topic_sub = nh.subscribe("NameOfTopic", 1000, writeMsgToLog);
	ros::spin();
}
