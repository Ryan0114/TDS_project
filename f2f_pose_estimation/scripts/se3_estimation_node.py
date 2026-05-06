#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf.transformations as tft

pose_pub = None
tf_pub = None


# ================================
# Lie algebra utilities
# ================================

def hat(w):
    """so(3) hat operator"""
    return np.array([
        [0, -w[2], w[1]],
        [w[2], 0, -w[0]],
        [-w[1], w[0], 0]
    ])


def se3_exp(xi):
    """Exponential map from se(3) to SE(3)"""
    rho = xi[:3]
    omega = xi[3:]

    theta = np.linalg.norm(omega)
    Omega = hat(omega)

    if theta < 1e-8:
        R = np.eye(3) + Omega
        V = np.eye(3)
    else:
        A = np.sin(theta) / theta
        B = (1 - np.cos(theta)) / (theta**2)
        C = (theta - np.sin(theta)) / (theta**3)

        R = np.eye(3) + A * Omega + B * (Omega @ Omega)
        V = np.eye(3) + B * Omega + C * (Omega @ Omega)

    t = V @ rho

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ================================
# Gauss-Newton on SE(3) (FIXED)
# ================================

def se3_gauss_newton(XA, XB, max_iter=10, damping=1e-6):
    """
    XA -> XB
    Solve: q ≈ T p
    """

    T = np.eye(4)

    for _ in range(max_iter):

        H = np.zeros((6, 6))
        b = np.zeros(6)

        R = T[:3, :3]
        t = T[:3, 3]

        for p, q in zip(XA, XB):

            # transform
            x = R @ p + t

            # residual
            r = q - x

            # ===== FIXED Jacobian =====
            # consistent with left perturbation:
            # x' = x + δρ + δω × x
            # r = q - x
            # dr = -δρ - δω × x
            J = np.zeros((3, 6))
            J[:, :3] = -np.eye(3)
            J[:, 3:] = -hat(x)

            H += J.T @ J
            b += J.T @ r

        # damping (Levenberg–Marquardt)
        H += damping * np.eye(6)

        # solve
        try:
            delta_xi = np.linalg.solve(H, b)
        except np.linalg.LinAlgError:
            rospy.logwarn("Singular system in SE(3) GN")
            break

        # step clipping (stability)
        norm = np.linalg.norm(delta_xi)
        if norm > 1.0:
            delta_xi *= 1.0 / norm

        # convergence
        if norm < 1e-6:
            break

        # left update (consistent)
        T = se3_exp(delta_xi) @ T

    return T


# ================================
# ROS callback
# ================================

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

    try:
        T = se3_gauss_newton(XA, XB)
    except Exception as e:
        rospy.logwarn(f"Optimization failed: {e}")
        return

    R = T[:3, :3]
    t = T[:3, 3]

    q = tft.quaternion_from_matrix(T)

    # =========================
    # Pose output
    # =========================
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

    # =========================
    # TF output
    # =========================
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

    rospy.loginfo(f"[SE3 GN FIXED] {len(XA)} correspondences")


# ================================
# main
# ================================

def main():
    global pose_pub, tf_pub

    rospy.init_node('se3_estimation_node')

    pose_pub = rospy.Publisher('/relative_pose', PoseStamped, queue_size=10)
    tf_pub = rospy.Publisher('/relative_transform', TransformStamped, queue_size=10)

    rospy.Subscriber('/feature_3d_matches', Float64MultiArray,
                     callback, queue_size=10)

    rospy.spin()


if __name__ == '__main__':
    main()
