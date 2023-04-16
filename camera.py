import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    roi = frame[50:200, 160:310]
    cv2.imshow('roi', roi)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)

    cv2.imshow('roi scaled and gray', roi)
    copy = frame.copy()
    cv2.rectangle(copy, (320, 100), (620, 400), (255, 0, 0), 5)
    roi = roi.reshape(1, 28, 28, 1)
    result = str(model.predict_classes(roi, 1, verbose=0)[0])
    cv2.putText(copy, getLetter(result), (300, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
    cv2.imshow('frame', copy)
    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break
cap.release()
cv2.destroyAllWindows()
