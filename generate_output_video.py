import cv2
import numpy as np

frames = np.load("frames.npy")

for frame in frames:
    cv2.imshow("Emojify", frame)
    cv2.waitKey(20)

