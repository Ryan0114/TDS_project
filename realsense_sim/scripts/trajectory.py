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

        self.model_name = rospy.get_param("~model_name", "d435i_camera")
        self.rate_hz = rospy.get_param("~rate", 30.0)

        # Trajectory parameters
        self.R = rospy.get_param("~radius", 2.0)
        self.omega = rospy.get_param("~omega", 0.3)
        self.height = rospy.get_param("~height", 1.0)

        self.pitch_amp = rospy.get_param("~pitch_amplitude", 0.4)  # radians
        self.pitch_freq = rospy.get_param("~pitch_frequency", 1.0)

        self.br = tf2_ros.TransformBroadcaster()

        rospy.wait_for_service("/gazebo/set_model_state")
        self.set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        self.rate = rospy.Rate(self.rate_hz)
        self.t0 = rospy.Time.now().to_sec()

    def run(self):
        while not rospy.is_shutdown():
            t = rospy.Time.now().to_sec() - self.t0

            # -----------------------------
            # 1. Circular motion (x-y plane)
            # -----------------------------
            x = self.R * math.cos(self.omega * t)
            y = self.R * math.sin(self.omega * t)
            z = self.height

            # -----------------------------
            # 2. Sinusoidal pitch motion
            # -----------------------------
            pitch = self.pitch_amp * math.sin(self.pitch_freq * t)
            yaw = 0.0
            roll = 0.0

            q = self.euler_to_quaternion(roll, pitch, yaw)

            # -----------------------------
            # 3. Move Gazebo model
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
                rospy.logwarn(f"SetModelState failed: {e}")

            # -----------------------------
            # 4. Publish TF
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
    def euler_to_quaternion(roll, pitch, yaw):
        """
        Convert RPY → quaternion
        """
        q = geometry_msgs.msg.Quaternion()

        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q.w = cr * cp * cy + sr * sp * sy
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy

        return q


if __name__ == "__main__":
    try:
        node = TrajectoryNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
