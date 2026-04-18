#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge

bridge = CvBridge()
prev_img = None
has_prev_img = False

orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

pub = None


def image_callback(msg):
    global prev_img, has_prev_img, pub

    curr_img = bridge.imgmsg_to_cv2(msg, 'bgr8')

    if has_prev_img:
        S = compute_similarity(prev_img, curr_img)
        rospy.loginfo(f"Similarity: {S}")
        pub.publish(S)

    prev_img = curr_img
    has_prev_img = True


def compute_similarity(imgA, imgB):
    kpA, desA = orb.detectAndCompute(imgA, None)
    kpB, desB = orb.detectAndCompute(imgB, None)

    if desA is None or desB is None:
        return 0.0

    matches = bf.knnMatch(desA, desB, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good) < 10:
        return 0.0

    ptsA = np.float32([kpA[m.queryIdx].pt for m in good])
    ptsB = np.float32([kpB[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 5.0)

    if mask is None:
        return 0.0

    return float(mask.ravel().sum()) / len(good)


def main():
    global pub

    rospy.init_node('image_sub', anonymous=True)

    pub = rospy.Publisher('/image_similarity', Float64, queue_size=10)

    rospy.Subscriber('/camera/color/image_raw', Image, image_callback, queue_size=10)

    rospy.spin()


if __name__ == '__main__':
    main()
