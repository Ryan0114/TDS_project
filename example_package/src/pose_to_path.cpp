#include <ros/ros.h>
#include <turtlesim/Pose.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseStamped.h>

class PoseToPath
{
public:
    PoseToPath()
    {
        pub_ = nh_.advertise<nav_msgs::Path>("turtle_path", 10);
        sub_ = nh_.subscribe("/turtle1/pose", 10, &PoseToPath::callback, this);

        path_.header.frame_id = "world"; // turtlesim frame
    }

    void callback(const turtlesim::Pose::ConstPtr& msg)
    {
        geometry_msgs::PoseStamped pose;
        pose.header.stamp = ros::Time::now();
        pose.header.frame_id = "world";

        pose.pose.position.x = msg->x;
        pose.pose.position.y = msg->y;
        pose.pose.position.z = 0.0;

        pose.pose.orientation.w = 1.0;

        path_.header.stamp = pose.header.stamp;
        path_.poses.push_back(pose);

        pub_.publish(path_);
	ros::Duration(0.05).sleep();
    }

private:
    ros::NodeHandle nh_;
    ros::Publisher pub_;
    ros::Subscriber sub_;
    nav_msgs::Path path_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pose_to_path");
    PoseToPath node;
    ros::spin();
}
