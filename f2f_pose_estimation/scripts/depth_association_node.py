#!/usr/bin/env python3

import rospy
import numpy as np

from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

from cv_bridge import CvBridge

import message_filters

from f2f_pose_estimation.msg import FeatureMatches


bridge = CvBridge()

pub = None

prev_depth = None
has_prev_depth = False


# aligned depth intrinsics
fx = 611.326416015625
fy = 610.279541015625
cx = 324.63714599609375
cy = 240.67333984375


def backproject(u, v, Z):

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    return np.array([X, Y, Z], dtype=np.float64)


def callback(match_msg, depth_msg):

    global prev_depth
    global has_prev_depth

    curr_depth = bridge.imgmsg_to_cv2(
        depth_msg,
        desired_encoding='passthrough'
    )

    if curr_depth.dtype == np.uint16:
        curr_depth = curr_depth.astype(np.float32) / 1000.0

    if not has_prev_depth:
        prev_depth = curr_depth.copy()
        has_prev_depth = True
        return

    data = np.array(match_msg.data, dtype=np.float64)

    if len(data) % 4 != 0:
        rospy.logwarn("Invalid match format")
        return

    matches = data.reshape(-1, 4)

    h, w = curr_depth.shape

    points_3d = []

    valid = 0

    for (uA, vA, uB, vB) in matches:

        uA_i = int(round(uA))
        vA_i = int(round(vA))

        uB_i = int(round(uB))
        vB_i = int(round(vB))

        if not (
            0 <= uA_i < w and
            0 <= vA_i < h and
            0 <= uB_i < w and
            0 <= vB_i < h
        ):
            continue

        ZA = prev_depth[vA_i, uA_i]
        ZB = curr_depth[vB_i, uB_i]

        if (
            ZA <= 0 or
            ZB <= 0 or
            np.isnan(ZA) or
            np.isnan(ZB)
        ):
            continue

        XA = backproject(uA, vA, ZA)
        XB = backproject(uB, vB, ZB)

        points_3d.append(
            np.hstack([XA, XB])
        )

        valid += 1

    prev_depth = curr_depth.copy()

    if len(points_3d) < 5:
        rospy.logwarn("Not enough valid 3D points")
        return

    points_3d = np.array(points_3d)

    msg_out = Float64MultiArray()

    msg_out.data = points_3d.flatten().tolist()

    pub.publish(msg_out)

    rospy.loginfo(
        f"Published {len(points_3d)} 3D correspondences"
    )


def main():

    global pub

    rospy.init_node('depth_association_node')

    pub = rospy.Publisher(
        '/feature_3d_matches',
        Float64MultiArray,
        queue_size=10
    )

    match_sub = message_filters.Subscriber(
        '/feature_matches',
        FeatureMatches
    )

    depth_sub = message_filters.Subscriber(
        '/camera/aligned_depth_to_color/image_raw',
        Image
    )

    ts = message_filters.ApproximateTimeSynchronizer(
        [match_sub, depth_sub],
        queue_size=20,
        slop=0.1
    )

    ts.registerCallback(callback)

    rospy.spin()


if __name__ == '__main__':
    main()
