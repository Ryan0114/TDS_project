#include <ros/ros.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>

#include <tf2_ros/transform_broadcaster.h>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Point3.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

class PGONode {

private:

    ros::NodeHandle nh_;
    ros::Subscriber sub_;

    tf2_ros::TransformBroadcaster br_;

    gtsam::ISAM2 isam_;
    gtsam::NonlinearFactorGraph graph_;
    gtsam::Values initial_;
    gtsam::Values result_;

    bool initialized_ = false;
    size_t k_ = 0;

    gtsam::Pose3 prev_pose_;

    gtsam::noiseModel::Diagonal::shared_ptr prior_noise_;
    gtsam::noiseModel::Diagonal::shared_ptr odom_noise_;

public:

    PGONode() {

        gtsam::ISAM2Params params;
        params.relinearizeThreshold = 0.1;
        params.relinearizeSkip = 1;
        isam_ = gtsam::ISAM2(params);

        // ---------------------------
        // CRITICAL: strong gauge fixing
        // ---------------------------
        gtsam::Vector6 prior_sigmas;
        prior_sigmas << 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6;

        gtsam::Vector6 odom_sigmas;
        odom_sigmas << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1;

        prior_noise_ = gtsam::noiseModel::Diagonal::Sigmas(prior_sigmas);
        odom_noise_  = gtsam::noiseModel::Diagonal::Sigmas(odom_sigmas);

        sub_ = nh_.subscribe(
            "/relative_pose",
            100,
            &PGONode::callback,
            this
        );

        ROS_INFO("PGO node initialized (stable version).");
    }

    gtsam::Pose3 toPose(const geometry_msgs::Pose& p) {

        return gtsam::Pose3(
            gtsam::Rot3::Quaternion(
                p.orientation.w,
                p.orientation.x,
                p.orientation.y,
                p.orientation.z
            ),
            gtsam::Point3(
                p.position.x,
                p.position.y,
                p.position.z
            )
        );
    }

    void callback(const geometry_msgs::PoseStamped::ConstPtr& msg) {

        ROS_INFO("PGO callback triggered");

        gtsam::Pose3 rel = toPose(msg->pose);

        size_t i = k_;
        size_t j = k_ + 1;

        // ---------------------------
        // FIRST POSE: FIX WORLD FRAME
        // ---------------------------
        if (!initialized_) {

            graph_.add(
                gtsam::PriorFactor<gtsam::Pose3>(
                    0,
                    gtsam::Pose3(),
                    prior_noise_
                )
            );

            initial_.insert(0, gtsam::Pose3());
            prev_pose_ = gtsam::Pose3();

            initialized_ = true;
            k_ = 1;

            return;
        }

        // ---------------------------
        // ensure safe initial guess
        // ---------------------------
        if (!initial_.exists(i)) {
            initial_.insert(i, prev_pose_);
        }

        gtsam::Pose3 predicted = prev_pose_.compose(rel);

        if (!initial_.exists(j)) {
            initial_.insert(j, predicted);
        }

        // ---------------------------
        // add odometry factor
        // ---------------------------
        graph_.add(
            gtsam::BetweenFactor<gtsam::Pose3>(
                i, j, rel, odom_noise_
            )
        );

        prev_pose_ = predicted;
        k_++;

        optimize(j);
    }

    void optimize(size_t latest) {

        isam_.update(graph_, initial_);

        graph_.resize(0);
        initial_.clear();

        result_ = isam_.calculateEstimate();

        if (result_.empty() || !result_.exists(latest)) {
            ROS_WARN("PGO: missing estimate");
            return;
        }

        gtsam::Pose3 p = result_.at<gtsam::Pose3>(latest);

        auto t = p.translation();
        auto q = p.rotation().toQuaternion();

        ROS_INFO_STREAM(
            "PGO: "
            << t.x() << ", "
            << t.y() << ", "
            << t.z()
        );

        geometry_msgs::TransformStamped tf;

        tf.header.stamp = ros::Time::now();
        tf.header.frame_id = "map";
        tf.child_frame_id = "camera";

        tf.transform.translation.x = t.x();
        tf.transform.translation.y = t.y();
        tf.transform.translation.z = t.z();

        tf.transform.rotation.x = q.x();
        tf.transform.rotation.y = q.y();
        tf.transform.rotation.z = q.z();
        tf.transform.rotation.w = q.w();

        br_.sendTransform(tf);
    }
};

int main(int argc, char** argv) {

    ros::init(argc, argv, "pgo_node");
    PGONode node;
    ros::spin();
}
