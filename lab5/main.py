import cv2 as cv
import numpy as np

# 1.
img1 = cv.imread("flower.jpg")

# 2.1.
cv.rectangle(img1, (50, 50), (180, 150), (0, 255, 0), 1)

# 2.2.
cv.rectangle(img1, (50, 100), (200, 200), (255, 0, 0), 2)

# 2.3.
cv.putText(img1, "OpenCV Task", (30, 220), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

# 2.4.
cv.imshow("img1", img1)

# 3.
img2 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

# 4.
img3 = cv.cvtColor(img1, cv.COLOR_BGR2LAB)

# 5.
img4 = img3[:, :, 0]

# 6.
cv.imshow("img2", img2)
cv.imshow("img4", img4)
cv.waitKey(500)

# 7. Безопасные вырезки
green_cut = img1[50:150, 30:180]
blue_cut = img1[100:200, 50:200]
img5 = np.vstack((green_cut, blue_cut))
cv.imshow("img5", img5)

# 8.
img6 = img1 / 255.0

# 9.
img7 = (img6 * 255).astype(np.uint8)

# 10.
cv.imshow("img7", img7)
cv.imwrite("final_img.jpg", img7)

# 11.
h, w = img7.shape[:2]
resized = cv.resize(img7, (w * 2, h * 3))
cv.imshow("resized", resized)

# 12.
img8 = cv.cvtColor(img3, cv.COLOR_LAB2BGR)
cv.imshow("img8", img8)

# 13.
pts = np.array([[80, 180], [110, 140], [140, 180], [110, 210]], np.int32).reshape((-1, 1, 2))
cv.polylines(img1, [pts], isClosed=True, color=(255, 255, 0), thickness=1)
cv.imshow("img1_poly", img1)

# 14.
res = cv.pointPolygonTest(pts, (110, 180), False)
print("Point inside polygon:", res > 0)

# 15.
def coords(event, x, y, flags, param):
    if event == cv.EVENT_MOUSEMOVE:
        print("Координати:", x, y)

cv.setMouseCallback("img1_poly", coords)

# 16.
cap = cv.VideoCapture("cat.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    cv.imshow("video", frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break
cap.release()

# 17.
cap2 = cv.VideoCapture("cat.mp4")
while cap2.isOpened():
    ret, frame = cap2.read()
    if not ret:
        break
    cv.rectangle(frame, (100, 100), (200, 200), (0, 0, 255), 2)
    cv.imshow("video_with_rect", frame)
    if cv.waitKey(20) & 0xFF == ord('q'):
        break
cap2.release()
cv.destroyAllWindows()