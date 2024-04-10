import numpy as np
import cv2 as cv

import imageio

gif = []

cap = cv.VideoCapture('video/traffic_smaller_trim.mp4')

ret, frame = cap.read()

prvs = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

hsv = np.zeros_like(frame)
hsv[..., 1] = 255

flow = None

while 1:
    ret, frame = cap.read()
    if not ret:
        break

    next = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs,
                                       next,
                                       flow,
                                       0.5,
                                       3,
                                       15,
                                       3,
                                       5,
                                       1.2,
                                       0)

    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    cv.imshow('frame', bgr)
    gif.append(cv.cvtColor(hsv, cv.COLOR_HSV2RGB))

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    prvs = next

imageio.mimsave('./images/dense.gif', gif)

cv.destroyAllWindows()