import cv2
import mediapipe as mp
import time
import module as htm
import serial


arduino = serial.Serial('COM3', 9600, timeout=1)
time.sleep(2)

pTime = 0
cTime = 0
cap = cv2.VideoCapture(1)
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    fingers = detector.fingersUp()
    count = fingers.count(1)
    arduino.write(str(count).encode())

    cv2.putText(img, f"Count: {count}", (10, 100),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.imshow("Image", img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()