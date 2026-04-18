'''
import rospy
import tf2_ros
import geometry_msgs.msg
import numpy as np
import math

rospy.init_node("trajectory")

br = tf2_ros.TransformBroadcaster()
rate = rospy.Rate(30)

t0 = rospy.Time.now().to_sec()

while not rospy.is_shutdown():
    t = rospy.Time.now().to_sec() - t0

    x = 0.5 * t
    y = 0.2 * math.sin(0.5 * t)
    z = 1.0 + 0.05 * math.sin(2*t)

    q = geometry_msgs.msg.Quaternion()
    q.w = 1.0

    tf = geometry_msgs.msg.TransformStamped()
    tf.header.stamp = rospy.Time.now()
    tf.header.frame_id = "world"
    tf.child_frame_id = "camera_link"

    tf.transform.translation.x = x
    tf.transform.translation.y = y
    tf.transform.translation.z = z
    tf.transform.rotation = q

    br.sendTransform(tf)
    rate.sleep()
    '''
    
#!/usr/bin/env python3

import rospy
import math
import tf2_ros
import geometry_msgs.msg

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState


class TrajectoryNode:
    def __init__(self):
        rospy.init_node("trajectory_node")

        # Parameters
        self.model_name = rospy.get_param("~model_name", "d435i_camera")
        self.rate_hz = rospy.get_param("~rate", 30.0)

        # TF broadcaster
        self.br = tf2_ros.TransformBroadcaster()

        # Wait for Gazebo service
        rospy.loginfo("Waiting for /gazebo/set_model_state...")
        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        rospy.loginfo("Connected to /gazebo/set_model_state")

        self.rate = rospy.Rate(self.rate_hz)
        self.t0 = rospy.Time.now().to_sec()

    def run(self):
        while not rospy.is_shutdown():
            t = rospy.Time.now().to_sec() - self.t0

            # -----------------------------
            # Trajectory definition (SE(3))
            # -----------------------------
            # Forward motion + slight lateral + vertical oscillation
            x = 0.5 * t
            y = 0.2 * math.sin(0.5 * t)
            z = 1.0 + 0.05 * math.sin(2.0 * t)

            # Yaw rotation (simulate turning while walking)
            yaw = 0.2 * math.sin(0.3 * t)

            # Convert yaw to quaternion (Z-axis rotation)
            q = self.yaw_to_quaternion(yaw)

            # -----------------------------
            # 1. Move Gazebo model
            # -----------------------------
            state = ModelState()
            state.model_name = self.model_name

            state.pose.position.x = x
            state.pose.position.y = y
            state.pose.position.z = z

            state.pose.orientation.x = q.x
            state.pose.orientation.y = q.y
            state.pose.orientation.z = q.z
            state.pose.orientation.w = q.w

            state.reference_frame = "world"

            try:
                self.set_state(state)
            except rospy.ServiceException as e:
                rospy.logwarn(f"Failed to set model state: {e}")

            # -----------------------------
            # 2. Publish TF (ground truth)
            # -----------------------------
            tf_msg = geometry_msgs.msg.TransformStamped()
            tf_msg.header.stamp = rospy.Time.now()
            tf_msg.header.frame_id = "world"
            tf_msg.child_frame_id = "camera_link"

            tf_msg.transform.translation.x = x
            tf_msg.transform.translation.y = y
            tf_msg.transform.translation.z = z

            tf_msg.transform.rotation = q

            self.br.sendTransform(tf_msg)

            self.rate.sleep()

    @staticmethod
    def yaw_to_quaternion(yaw):
        """
        Convert yaw angle (Z-axis rotation) to quaternion
        """
        q = geometry_msgs.msg.Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q


if __name__ == "__main__":
    try:
        node = TrajectoryNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
