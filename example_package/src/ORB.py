import cv2
import matplotlib.pyplot as plt
import numpy as np

def compute_similarity(imgA, imgB):
    orb = cv2.ORB_create(2000)

    kpA, desA = orb.detectAndCompute(imgA, None)
    kpB, desB = orb.detectAndCompute(imgB, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(desA, desB, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 10:
        return 0

    ptsA = np.float32([kpA[m.queryIdx].pt for m in good])
    ptsB = np.float32([kpB[m.trainIdx].pt for m in good])

    H, mask = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 5.0)

    if mask is None:
        return 0

    inliers = mask.ravel().sum()

    return inliers / len(good)

def show_image(title, img, cmap=None):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.show()

# -----------------------------
# 1. Load images (grayscale)
# -----------------------------
img1 = cv2.imread("book_1.png", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("book_2.png", cv2.IMREAD_GRAYSCALE)

if img1 is None or img2 is None:
    raise FileNotFoundError("Could not load images. Check file paths.")

# -----------------------------
# 2. ORB detector
# -----------------------------
orb = cv2.ORB_create(nfeatures=1000)

# Detect keypoints + descriptors
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# -----------------------------
# 3. Visualize keypoints
# -----------------------------
img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)

show_image("Image 1 Keypoints", img1_kp)
show_image("Image 2 Keypoints", img2_kp)

# -----------------------------
# 4. Feature matching (Hamming distance)
# -----------------------------
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1, des2)

# Sort matches by distance (lower = better)
matches = sorted(matches, key=lambda x: x.distance)

# -----------------------------
# 5. Draw top matches
# -----------------------------
matched_img = cv2.drawMatches(
    img1, kp1,
    img2, kp2,
    matches[:50],  # show top 50 matches
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

show_image("ORB Feature Matches", matched_img)

similarity_score = compute_similarity(img1, img2)
print("similarity:", similarity_score)

