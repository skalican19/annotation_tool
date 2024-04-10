from __future__ import print_function
import cv2
import imageio

backSub = cv2.createBackgroundSubtractorMOG2()

gif = []

video_path = 'video/KITTI-17-raw.webm'  # Update this path
capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(video_path))

if not capture.isOpened():
    print('Unable to open: ' + video_path)
    exit(0)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    fgMask = backSub.apply(frame, learningRate=0.025)

    cv2.imshow('Frame', frame)
    cv2.imshow('Background', backSub.getBackgroundImage())
    cv2.imshow('FG Mask', fgMask)

    gif.append(fgMask)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

imageio.mimsave('./images/mog2.gif', gif)
cv2.imwrite('./images/mog2_bg.png', backSub.getBackgroundImage())
