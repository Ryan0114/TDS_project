#!/usr/bin/env python3

import rospy
import numpy as np

from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped, TransformStamped

import tf.transformations as tft
import tf2_ros


pose_pub = None
tf_pub = None
tf_broadcaster = None


def umeyama_alignment(XA, XB):
    """
    Solve XB ≈ R XA + t
    XA, XB: (N,3)
    """

    assert XA.shape == XB.shape

    N = XA.shape[0]

    mu_A = np.mean(XA, axis=0)
    mu_B = np.mean(XB, axis=0)

    A = XA - mu_A
    B = XB - mu_B

    H = (A.T @ B) / N

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = mu_B - R @ mu_A

    return R, t


def callback(msg):

    global pose_pub, tf_pub, tf_broadcaster

    data = np.array(msg.data, dtype=np.float64)

    if len(data) % 6 != 0:
        rospy.logwarn("Invalid 3D correspondence format")
        return

    pts = data.reshape(-1, 6)

    if len(pts) < 5:
        rospy.logwarn("Not enough correspondences")
        return

    XA = pts[:, 0:3]
    XB = pts[:, 3:6]

    try:
        R, t = umeyama_alignment(XA, XB)
    except Exception as e:
        rospy.logwarn(f"SE3 estimation failed: {e}")
        return

    # ----------------------------
    # pose construction
    # ----------------------------
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    q = tft.quaternion_from_matrix(T)

    # ----------------------------
    # publish PoseStamped
    # ----------------------------
    pose_msg = PoseStamped()
    pose_msg.header.stamp = rospy.Time.now()
    pose_msg.header.frame_id = "frame_A"

    pose_msg.pose.position.x = float(t[0])
    pose_msg.pose.position.y = float(t[1])
    pose_msg.pose.position.z = float(t[2])

    pose_msg.pose.orientation.x = float(q[0])
    pose_msg.pose.orientation.y = float(q[1])
    pose_msg.pose.orientation.z = float(q[2])
    pose_msg.pose.orientation.w = float(q[3])

    pose_pub.publish(pose_msg)

    # ----------------------------
    # TF publish
    # ----------------------------
    tf_msg = TransformStamped()
    tf_msg.header = pose_msg.header
    tf_msg.header.frame_id = "frame_A"
    tf_msg.child_frame_id = "frame_B"

    tf_msg.transform.translation.x = float(t[0])
    tf_msg.transform.translation.y = float(t[1])
    tf_msg.transform.translation.z = float(t[2])

    tf_msg.transform.rotation.x = float(q[0])
    tf_msg.transform.rotation.y = float(q[1])
    tf_msg.transform.rotation.z = float(q[2])
    tf_msg.transform.rotation.w = float(q[3])

    tf_pub.publish(tf_msg)

    if tf_broadcaster is not None:
        tf_broadcaster.sendTransform(tf_msg)

    # ----------------------------
    # dominant motion analysis
    # ----------------------------

    tx, ty, tz = map(float, t)
    abs_t = np.abs([tx, ty, tz])

    dominant_idx = np.argmax(abs_t)
    axis_names = ['X', 'Y', 'Z']
    dominant_axis = axis_names[dominant_idx]
    dominant_value = [tx, ty, tz][dominant_idx]

    direction = "+" if dominant_value >= 0 else "-"

    semantic = {
        'X+': 'right',
        'X-': 'left',
        'Y+': 'down',
        'Y-': 'up',
        'Z+': 'forward',
        'Z-': 'backward'
    }

    semantic_label = semantic.get(dominant_axis + direction, '')

    # ----------------------------
    # ANSI colors
    # ----------------------------
    RESET = "\033[0m"
    BOLD = "\033[1m"

    RED = "\033[91m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"

    color_map = {
        'X': RED,
        'Y': GREEN,
        'Z': BLUE
    }

    motion_threshold = 0.02  # meters

    if abs(dominant_value) < motion_threshold:
        dominant_text = "STATIONARY"
        color = RESET
        magnitude_text = 0.0
    else:
        dominant_text = f"{direction}{dominant_axis} ({semantic_label})"
        color = color_map[dominant_axis]
        magnitude_text = abs(dominant_value)

    # ----------------------------
    # orientation
    # ----------------------------

    roll, pitch, yaw = tft.euler_from_matrix(T)

    rospy.loginfo(
        "\n"
        "=====================================\n"
        f"{BOLD}{color}"
        f"DOMINANT MOTION: {dominant_text}\n"
        f"Magnitude: {magnitude_text:.4f} m"
        f"{RESET}\n"
        "-------------------------------------\n"
        f"x = {tx:.4f} m\n"
        f"y = {ty:.4f} m\n"
        f"z = {tz:.4f} m\n"
        "-------------------------------------\n"
        f"roll  = {np.degrees(roll):.2f} deg\n"
        f"pitch = {np.degrees(pitch):.2f} deg\n"
        f"yaw   = {np.degrees(yaw):.2f} deg\n"
        "====================================="
    )


def main():

    global pose_pub, tf_pub, tf_broadcaster

    rospy.init_node('se3_estimation_node')

    pose_pub = rospy.Publisher(
        '/relative_pose',
        PoseStamped,
        queue_size=10
    )

    tf_pub = rospy.Publisher(
        '/relative_transform',
        TransformStamped,
        queue_size=10
    )

    tf_broadcaster = tf2_ros.TransformBroadcaster()

    rospy.Subscriber(
        '/feature_3d_matches',
        Float64MultiArray,
        callback,
        queue_size=10
    )

    rospy.spin()


if __name__ == '__main__':
    main()
