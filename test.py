import cv2
# 替换为你的相机对应索引，通常是0或1
cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法获取图像，请检查相机连接或权限")
        break
    cv2.imshow("Camera Check", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
