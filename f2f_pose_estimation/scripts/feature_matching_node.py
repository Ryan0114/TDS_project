#!/usr/bin/env python3

import cv2
import numpy as np
import rospy

from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge

from f2f_pose_estimation.msg import FeatureMatches


bridge = CvBridge()

orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

prev_img = None
has_prev = False

sim_pub = None
match_pub = None


def compute_matches(imgA, imgB):

    kpA, desA = orb.detectAndCompute(imgA, None)
    kpB, desB = orb.detectAndCompute(imgB, None)

    if desA is None or desB is None:
        return np.empty((0, 4)), 0.0

    matches = bf.knnMatch(desA, desB, k=2)

    good = [
        m for m, n in matches
        if m.distance < 0.75 * n.distance
    ]

    if len(good) < 10:
        return np.empty((0, 4)), 0.0

    ptsA = np.float32([kpA[m.queryIdx].pt for m in good])
    ptsB = np.float32([kpB[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(
        ptsA,
        ptsB,
        cv2.RANSAC,
        5.0
    )

    if mask is None:
        return np.empty((0, 4)), 0.0

    mask = mask.ravel().astype(bool)

    ptsA = ptsA[mask]
    ptsB = ptsB[mask]

    matches_out = np.hstack([ptsA, ptsB])

    similarity = float(np.sum(mask)) / len(good)

    return matches_out, similarity


def image_callback(msg):

    global prev_img
    global has_prev

    curr_img = bridge.imgmsg_to_cv2(msg, 'bgr8')

    if has_prev:

        matches, similarity = compute_matches(
            prev_img,
            curr_img
        )

        sim_pub.publish(similarity)

        fm = FeatureMatches()

        # timestamp = CURRENT frame (B)
        fm.header = msg.header

        fm.data = matches.flatten().tolist()

        match_pub.publish(fm)

        rospy.loginfo(
            f"Published {len(matches)} feature matches"
        )

    prev_img = curr_img
    has_prev = True


def main():

    global sim_pub
    global match_pub

    rospy.init_node('feature_matching_node')

    sim_pub = rospy.Publisher(
        '/image_similarity',
        Float64,
        queue_size=10
    )

    match_pub = rospy.Publisher(
        '/feature_matches',
        FeatureMatches,
        queue_size=10
    )

    rospy.Subscriber(
        '/camera/color/image_raw',
        Image,
        image_callback,
        queue_size=10
    )

    rospy.spin()


if __name__ == '__main__':
    main()
