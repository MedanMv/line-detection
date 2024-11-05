import cv2
import numpy as np

def detect_stop_signs(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    stop_signs = stop_sign_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in stop_signs:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, 'Stop Sign', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    return frame

def determinant(beginning, end, centerP):
    D = ((end[0] - beginning[0]) * (centerP[1] - beginning[1])) - ((end[1] - beginning[1]) * (centerP[0] - beginning[0]))
    if abs(end[0] - centerP[0]) <= 40:
        return "forward"
    elif D > 0:
        return "turn right"
    elif D < 0:
        return "turn left"
    else:
        return "forward"

def find_avg_point(p1, p2):
    x = int((p1[0] + p2[0]) / 2)
    y = int((p1[1] + p2[1]) / 2)
    return (x, y)

def nothing(x):
    pass

stop_sign_cascade = cv2.CascadeClassifier('B:/eek/PY/3/stop_sign_classifier_2.xml')

cap = cv2.VideoCapture(0)

kernel1 = np.ones((3, 3), np.uint8) 
kernel2 = np.ones((7, 7), np.uint8)

cv2.namedWindow("2_Points")

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.resize(img, (600, 400))
    
    height, width, _ = img.shape
    
    start_row = height // 2
    end_row = height

    road_frame = img[start_row:end_row, :width]

    centreX = int(0.5 * img.shape[1])
    centerY = int(img.shape[0] // 3)
    center = (centreX, centerY)
    
    img_erosion = cv2.erode(road_frame, kernel1, 1) 
    img_dilation = cv2.dilate(img_erosion, kernel2, 1)
    blur = cv2.GaussianBlur(img_dilation, (21, 21), 3)
    thresh5_gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY) 
    _, thresh5 = cv2.threshold(thresh5_gray, 90, 255, cv2.THRESH_TOZERO_INV)
    
    contours, _ = cv2.findContours(thresh5, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_list = []

    print(len(contours))

    cv2.imshow('Processed Frame', thresh5)

    if len(contours) >= 1:
        for contour in contours:
            bottom_left = tuple(contour[np.argmin(contour[:, :, 1] + contour[:, :, 0])][0])
            bottom_right = tuple(contour[np.argmin(contour[:, :, 1] - contour[:, :, 0])][0])
            top_left = tuple(contour[np.argmax(contour[:, :, 1] - contour[:, :, 0])][0])
            top_right = tuple(contour[np.argmax(contour[:, :, 1] + contour[:, :, 0])][0])
            
            top_point = find_avg_point(top_right, top_left)
            bottom_point = find_avg_point(bottom_right, bottom_left)

            contours_list.append([top_point, bottom_point])

        if contours_list:
            cv2.circle(img, top_point, 5, (0, 0, 255), -1)
            cv2.circle(img, bottom_point, 5, (0, 255, 0), -1)
            cv2.circle(img, center, 5, (255, 255, 0), -1)
            cv2.line(img, bottom_point, top_point, (255, 0, 0), 2)  

            top_of_screen = (center[0], 0)
            cv2.line(img, center, top_of_screen, (0, 165, 255), 2)  

            direction = determinant(bottom_point, top_point, top_of_screen)
            print(direction)
        
    img = detect_stop_signs(img)

    cv2.imshow('2_Points', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
