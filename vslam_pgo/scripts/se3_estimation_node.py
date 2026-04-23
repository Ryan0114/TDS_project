#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf.transformations as tft

pose_pub = None
tf_pub = None


def umeyama_alignment(XA, XB):
    """
    XA, XB: shape (N,3)
    Returns R (3x3), t (3,)
    """

    assert XA.shape == XB.shape
    N = XA.shape[0]

    # --- centroids ---
    mu_A = np.mean(XA, axis=0)
    mu_B = np.mean(XB, axis=0)

    # --- center the points ---
    A_centered = XA - mu_A
    B_centered = XB - mu_B

    # --- covariance matrix ---
    H = A_centered.T @ B_centered / N

    # --- SVD ---
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # --- reflection correction ---
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    # --- translation ---
    t = mu_B - R @ mu_A

    return R, t


def callback(msg):
    global pose_pub, tf_pub

    data = np.array(msg.data, dtype=np.float64)

    if len(data) % 6 != 0:
        rospy.logwarn("Invalid 3D match message")
        return

    pts = data.reshape(-1, 6)

    if len(pts) < 5:
        rospy.logwarn("Not enough correspondences")
        return

    XA = pts[:, 0:3]
    XB = pts[:, 3:6]

    # --- estimate pose ---
    try:
        R, t = umeyama_alignment(XA, XB)
    except Exception as e:
        rospy.logwarn(f"SVD failed: {e}")
        return

    # --- convert to quaternion ---
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    q = tft.quaternion_from_matrix(T)

    # --- publish PoseStamped ---
    pose_msg = PoseStamped()
    pose_msg.header.stamp = rospy.Time.now()
    pose_msg.header.frame_id = "camera_color_optical_frame"

    pose_msg.pose.position.x = t[0]
    pose_msg.pose.position.y = t[1]
    pose_msg.pose.position.z = t[2]

    pose_msg.pose.orientation.x = q[0]
    pose_msg.pose.orientation.y = q[1]
    pose_msg.pose.orientation.z = q[2]
    pose_msg.pose.orientation.w = q[3]

    pose_pub.publish(pose_msg)

    # --- publish TransformStamped ---
    tf_msg = TransformStamped()
    tf_msg.header.stamp = pose_msg.header.stamp
    tf_msg.header.frame_id = "frame_A"
    tf_msg.child_frame_id = "frame_B"

    tf_msg.transform.translation.x = t[0]
    tf_msg.transform.translation.y = t[1]
    tf_msg.transform.translation.z = t[2]

    tf_msg.transform.rotation.x = q[0]
    tf_msg.transform.rotation.y = q[1]
    tf_msg.transform.rotation.z = q[2]
    tf_msg.transform.rotation.w = q[3]

    tf_pub.publish(tf_msg)

    rospy.loginfo(f"Published pose with {len(XA)} correspondences")


def main():
    global pose_pub, tf_pub

    rospy.init_node('se3_estimation_node')

    pose_pub = rospy.Publisher('/relative_pose', PoseStamped, queue_size=10)
    tf_pub = rospy.Publisher('/relative_transform', TransformStamped, queue_size=10)

    rospy.Subscriber('/feature_3d_matches', Float64MultiArray, callback, queue_size=10)

    rospy.spin()


if __name__ == '__main__':
    main()
