
from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

model = YOLO('best_wp.pt')

classNames = ['bearing']

while True:
    success, img = cap.read()
    print(success,img)
    img = cv2.resize(img, (640, 480))  # Adjust the size as needed
    result = model(img, stream=True)
    print(result)
    for i in result:
        print(i)
        box = i.boxes
        for b in box:
            x1,y1,x2,y2 = b.xyxy[0]
            x1, y1, x2, y2 =  int(x1), int(y1), int(x2), int(y2)
            #print(x1, y1, x2, y2)
            cv2.rectangle(img, (x1,y1),(x2,y2),(255,0,255),2)

            # diameter calculation
            width = x2 - x1
            #print(width)
            # 14 mm = 53 px
            mm = width * 0.264583
            # ratio_px_mm = 53/14
            # mm = width / ratio_px_mm
            # cm = mm/10
            #print(mm)
            # print(cm)
            # confidence
            conf = math.ceil((box.conf[0]*100))/100
            #print(conf)

            # class name
            cls = int(box.cls[0])

            cvzone.putTextRect(img, f'{classNames[cls]} {mm}MM', (max(0, x1), max(0, y1)), scale=1, thickness=1)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()



