#include "ros/ros.h"
#include "std_msgs/String.h"
#include "example_package/MyMessage.h"

int main(int argc, char** argv) {
	ros::init(argc, argv, "publisher");
	ros::NodeHandle nh;

	ros::Publisher topic_pub = nh.advertise<std_msgs::String>("NameOfTopic", 1000);
	ros::Publisher profile_pub = nh.advertise<example_package::MyMessage>("profile", 1000);
	ros::Rate loop_rate(1);
	
	while(ros::ok()) {
		std_msgs::String msg;
		msg.data = "Hello World!";
		
		example_package::MyMessage msg2;
		msg2.age = 21;
		msg2.name = "Ryan";

		topic_pub.publish(msg);
		profile_pub.publish(msg2);
		ros::spinOnce();
		loop_rate.sleep();
	}

	return 0;
}
