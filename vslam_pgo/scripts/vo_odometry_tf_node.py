#!/usr/bin/env python3
import rospy
import numpy as np
import tf2_ros
import geometry_msgs.msg
from geometry_msgs.msg import TransformStamped

import tf.transformations as tft


class VOIntegrator:
    def __init__(self):
        self.br = tf2_ros.TransformBroadcaster()

        # world → camera pose
        self.T_wc = np.eye(4)

        self.last_time = None

        rospy.Subscriber('/relative_transform',
                         TransformStamped,
                         self.callback,
                         queue_size=100)

    def transform_to_matrix(self, tmsg):
        t = tmsg.transform.translation
        q = tmsg.transform.rotation

        T = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
        T[0, 3] = t.x
        T[1, 3] = t.y
        T[2, 3] = t.z
        return T

    def callback(self, msg):
        # --- relative transform: camera_k-1 → camera_k ---
        T_k = self.transform_to_matrix(msg)

        # --- integrate: world → camera_k ---
        self.T_wc = self.T_wc @ T_k

        # --- extract translation + rotation ---
        t = self.T_wc[:3, 3]
        q = tft.quaternion_from_matrix(self.T_wc)

        # --- publish TF ---
        tf_msg = TransformStamped()
        tf_msg.header.stamp = msg.header.stamp
        tf_msg.header.frame_id = "world"
        tf_msg.child_frame_id = "camera"

        tf_msg.transform.translation.x = t[0]
        tf_msg.transform.translation.y = t[1]
        tf_msg.transform.translation.z = t[2]

        tf_msg.transform.rotation.x = q[0]
        tf_msg.transform.rotation.y = q[1]
        tf_msg.transform.rotation.z = q[2]
        tf_msg.transform.rotation.w = q[3]

        self.br.sendTransform(tf_msg)

        rospy.loginfo(f"Pose: {t}")


def main():
    rospy.init_node('vo_tf_broadcaster_node')
    VOIntegrator()
    rospy.spin()


if __name__ == '__main__':
    main()
