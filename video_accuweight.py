import imageio
import cv2
import numpy as np

video_path = 'video/KITTI-17-raw.webm'
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
if not ret:
    print("Can't receive frame (stream end?). Exiting ...")
    exit()

gif = []

background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
background = cv2.GaussianBlur(background, (11, 11), 0)

background = np.float32(background)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)

    cv2.accumulateWeighted(gray, background, 0.01)

cap.release()
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(background))
    thresh = cv2.threshold(frameDelta, 100, 255, cv2.THRESH_BINARY)[1]

    # Display the current frame, background, and the foreground mask
    cv2.imshow('Frame', frame)
    cv2.imshow('Background', cv2.convertScaleAbs(background))
    cv2.imshow('Foreground Mask', thresh)

    gif.append(thresh)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

imageio.mimsave('./images/accuweight.gif', gif)
cv2.imwrite('./images/accuweight_bg.png', background)

cap.release()
cv2.destroyAllWindows()

