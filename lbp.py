from skimage import feature
import cv2
import numpy as np

def localbinarypattern(lbp):
    numPoints = 24
    radius = 3

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray, numPoints, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=range(0, numPoints + 3), range=(0, numPoints + 2))

    imean = np.mean(lbp)

    grayf = lbp.astype(np.float32)
    grayf2 = grayf * grayf
    imeanf2 = np.mean(grayf2)
    variance = imeanf2 - imean ** 2

    p = np.array([(lbp == v).sum() for v in range(256)])
    p = p / p.sum()
    entropy = -(p[p > 0] * np.log2(p[p > 0])).sum()

    return(imean, variance, entropy)
