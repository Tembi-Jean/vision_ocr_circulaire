#!/usr/bin/python3
# 2017.10.10 12:44:37 CST
# 2017.10.10 14:08:57 CST
import cv2
import numpy as np

##(1) Read and resize the original image(too big)
img = cv2.imread("imgbasler3.png")
H, W = img.shape[:2]
img = cv2.resize(img, (W//4, H//4))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Appliquer une égalisation d'histogramme pour améliorer le contraste
eq_gray = cv2.equalizeHist(gray)

## (2) Detect circles
circles = cv2.HoughCircles(gray, method=cv2.HOUGH_GRADIENT, dp=1, minDist=3, circles=None, param1=200, param2=100, minRadius = 200, maxRadius=0 )

## make canvas
canvas = img.copy()

## (3) Get the mean of centers and do offset
if circles is not None:
    circles = np.int0(np.array(circles))
    x,y,r = 0,0,0
    for ptx,pty, radius in circles[0]:
        cv2.circle(canvas, (ptx,pty), radius, (0,255,0), 1, 16)
        x += ptx
        y += pty
        r += radius
    cnt = len(circles[0])
    x = x//cnt
    y = y//cnt
    r = r//cnt
    x+=5
    y-=7

    ## (4) Draw the labels in red
    for r in range(100, r, 20):
        cv2.circle(canvas, (x,y), r, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.circle(canvas, (x,y), 3, (0,0,255), -1)

    ## (5) logPolar and rotate
    polar = cv2.logPolar(img, (int(x), int(y)), 80, cv2.WARP_FILL_OUTLIERS)
    rotated = cv2.rotate(polar, cv2.ROTATE_90_COUNTERCLOCKWISE)

    ## (6) Display the result
    cv2.imshow("Canvas", canvas)

    cv2.imshow("polar", polar)
    cv2.imshow("rotated", rotated)

    cv2.waitKey();cv2.destroyAllWindows()

else:
    print("No circles detected.")
