#include "ros/ros.h"
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <std_msgs/Float64MultiArray.h>
#include <tf2_ros/transform_broadcaster.h>

// GTSAM Core Headers
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

// GTSAM Factor Headers
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>

class PGONode {
private:
    ros::NodeHandle nh_;
    ros::Subscriber odom_sub_;
    ros::Subscriber loop_sub_;
    tf2_ros::TransformBroadcaster br_;

    // GTSAM Engine
    gtsam::ISAM2 isam_;
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values initial_values_;
    gtsam::Values latest_estimate_;

    size_t pose_count_ = 0;
    gtsam::Pose3 prev_pose_;

    // Noise Models
    gtsam::noiseModel::Diagonal::shared_ptr prior_noise_;
    gtsam::noiseModel::Diagonal::shared_ptr odom_noise_;

public:
    PGONode() {
        // Initialize ISAM2 parameters
        gtsam::ISAM2Params params;
        isam_ = gtsam::ISAM2(params);

        // Define Diagonal Noise Models (6-axis Vector: Roll, Pitch, Yaw, X, Y, Z)
        gtsam::Vector6 prior_sigmas, odom_sigmas;
        prior_sigmas << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;
        odom_sigmas  << 0.2, 0.2, 0.2, 0.2, 0.2, 0.2;

        prior_noise_ = gtsam::noiseModel::Diagonal::Sigmas(prior_sigmas);
        odom_noise_  = gtsam::noiseModel::Diagonal::Sigmas(odom_sigmas);

        // ROS Subscribers
        odom_sub_ = nh_.subscribe("/relative_pose", 10, &PGONode::odomCallback, this);
        loop_sub_ = nh_.subscribe("/loop_closure_constraint", 10, &PGONode::loopCallback, this);

        ROS_INFO("Pose Graph Optimization Node Initialized (C++).");
    }

    // Convert ROS PoseStamped to GTSAM Pose3
    gtsam::Pose3 poseToGtsam(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        const auto& p = msg->pose.position;
        const auto& q = msg->pose.orientation;
        
        return gtsam::Pose3(
            gtsam::Rot3::Quaternion(q.w, q.x, q.y, q.z),
            gtsam::Point3(p.x, p.y, p.z)
        );
    }

    // Odometry Callback
    void odomCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
        gtsam::Pose3 curr_pose = poseToGtsam(msg);
        size_t i = pose_count_;
        pose_count_++;

        initial_values_.insert(i, curr_pose);

        if (i == 0) {
            // Anchor the first node to world origin space
            graph_.add(gtsam::PriorFactor<gtsam::Pose3>(i, curr_pose, prior_noise_));
            prev_pose_ = curr_pose;
            return;
        }

        // Relative transform i-1 -> i
        gtsam::Pose3 rel = prev_pose_.between(curr_pose);
        graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(i - 1, i, rel, odom_noise_));

        prev_pose_ = curr_pose;
        optimize();
    }

    // Loop Closure Callback
    void loopCallback(const std_msgs::Float64MultiArray::ConstPtr& msg) {
        if (msg->data.size() != 13) {
            ROS_WARN("Invalid loop closure message array size. Expected 13.");
            return;
        }

        size_t i = static_cast<size_t>(msg->data[0]);
        size_t j = static_cast<size_t>(msg->data[1]);
        
        double dx = msg->data[2];
        double dy = msg->data[3];
        double dz = msg->data[4];
        
        double qx = msg->data[5];
        double qy = msg->data[6];
        double qz = msg->data[7];
        double qw = msg->data[8];

        gtsam::Pose3 rel(
            gtsam::Rot3::Quaternion(qw, qx, qy, qz),
            gtsam::Point3(dx, dy, dz)
        );

        graph_.add(gtsam::BetweenFactor<gtsam::Pose3>(i, j, rel, odom_noise_));
        optimize();
    }

    // Execute ISAM2 pass
    void optimize() {
        isam_.update(graph_, initial_values_);
        
        // Clear incremental structures
        graph_.resize(0);
        initial_values_.clear();

        latest_estimate_ = isam_.calculateEstimate();
        publishTF(latest_estimate_);
    }

    // TF Broadcaster
    void publishTF(const gtsam::Values& result) {
        if (pose_count_ == 0) return;

        size_t i = pose_count_ - 1;
        if (!result.exists(i)) return;

        gtsam::Pose3 pose = result.at<gtsam::Pose3>(i);
        gtsam::Point3 t = pose.translation();
        gtsam::Quaternion q = pose.rotation().toQuaternion();

        geometry_msgs::TransformStamped tf_msg;
        tf_msg.header.stamp = ros::Time::now();
        tf_msg.header.frame_id = "map";
        tf_msg.child_frame_id = "camera";

        tf_msg.transform.translation.x = t.x();
        tf_msg.transform.translation.y = t.y();
        tf_msg.transform.translation.z = t.z();

        tf_msg.transform.rotation.x = q.x();
        tf_msg.transform.rotation.y = q.y();
        tf_msg.transform.rotation.z = q.z();
        tf_msg.transform.rotation.w = q.w();

        br_.sendTransform(tf_msg);
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "pgo_node");
    PGONode node;
    ros::spin();
    return 0;
}
