#!/usr/bin/env python3

import rospy
import numpy as np
import cv2

from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import message_filters
import tf.transformations as tft


bridge = CvBridge()

# -----------------------------
# camera intrinsics
# -----------------------------
fx = 611.3264
fy = 610.2795
cx = 324.6371
cy = 240.6733


# -----------------------------
# keyframe storage
# -----------------------------
class FrameDB:

    def __init__(self):
        self.images = {}
        self.depths = {}
        self.kps = {}
        self.des = {}

        self.orb = cv2.ORB_create(2000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    def add(self, idx, img, depth):

        kp, des = self.orb.detectAndCompute(img, None)

        self.images[idx] = img
        self.depths[idx] = depth
        self.kps[idx] = kp
        self.des[idx] = des


db = FrameDB()

# -----------------------------
# backprojection
# -----------------------------
def backproject(u, v, Z):

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    return np.array([X, Y, Z], dtype=np.float64)


# -----------------------------
# SE3 estimation (REAL)
# -----------------------------
def estimate_se3(i, j):

    kp1, des1 = db.kps[i], db.des[i]
    kp2, des2 = db.kps[j], db.des[j]

    if des1 is None or des2 is None:
        return None

    matches = db.bf.knnMatch(des1, des2, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 20:
        return None

    pts_i = []
    pts_j = []

    depth_i = db.depths[i]
    depth_j = db.depths[j]

    h, w = depth_i.shape

    for m in good:

        u1, v1 = kp1[m.queryIdx].pt
        u2, v2 = kp2[m.trainIdx].pt

        u1, v1 = int(u1), int(v1)
        u2, v2 = int(u2), int(v2)

        if not (0 <= u1 < w and 0 <= v1 < h and
                0 <= u2 < w and 0 <= v2 < h):
            continue

        z1 = depth_i[v1, u1]
        z2 = depth_j[v2, u2]

        if z1 <= 0 or z2 <= 0:
            continue

        X1 = backproject(u1, v1, z1)
        X2 = backproject(u2, v2, z2)

        pts_i.append(X1)
        pts_j.append(X2)

    if len(pts_i) < 10:
        return None

    XA = np.array(pts_i)
    XB = np.array(pts_j)

    # SE3 (Umeyama)
    muA = XA.mean(axis=0)
    muB = XB.mean(axis=0)

    A = XA - muA
    B = XB - muB

    H = A.T @ B

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = muB - R @ muA

    return R, t


# -----------------------------
# loop closure trigger
# -----------------------------
def sim_callback(msg):

    i = sim_callback.idx
    sim_callback.idx += 1

    sim = msg.data

    db.add(i, sim_callback.last_img, sim_callback.last_depth)

    sim_callback.last_img = None
    sim_callback.last_depth = None

    # naive loop search
    for j in range(i-30):

        if j not in db.images:
            continue

        # simple heuristic: revisit candidates
        if abs(sim) > 0.75:

            result = estimate_se3(j, i)

            if result is None:
                continue

            R, t = result

            q = tft.quaternion_from_matrix(
                np.vstack([
                    np.hstack([R, t.reshape(3,1)]),
                    [0,0,0,1]
                ])
            )

            msg_out = Float64MultiArray()
            msg_out.data = [
                float(j), float(i),
                float(t[0]), float(t[1]), float(t[2]),
                float(q[0]), float(q[1]), float(q[2]), float(q[3])
            ]

            pub.publish(msg_out)

            rospy.loginfo(f"REAL loop closure: {j} ↔ {i}")

sim_callback.idx = 0
sim_callback.last_img = None
sim_callback.last_depth = None


def img_callback(msg):
    sim_callback.last_img = bridge.imgmsg_to_cv2(msg, "bgr8")


def depth_callback(msg):
    d = bridge.imgmsg_to_cv2(msg, "passthrough")
    if d.dtype == np.uint16:
        d = d.astype(np.float32)/1000.0
    sim_callback.last_depth = d


def main():

    global pub

    rospy.init_node("loop_closure_node")

    pub = rospy.Publisher(
        "/loop_closure_constraint",
        Float64MultiArray,
        queue_size=10
    )

    rospy.Subscriber("/camera/color/image_raw", Image, img_callback)
    rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, depth_callback)

    rospy.Subscriber("/image_similarity", Float64, sim_callback)

    rospy.spin()


if __name__ == "__main__":
    main()
