import cv2
import numpy as np
from tensorflow.keras.models import load_model
from symbol import Symbol

cap = cv2.VideoCapture(0)

_, frame = cap.read()
frame_shape = frame.shape
height = frame_shape[0]
width = frame_shape[1]
print (frame_shape)

model = load_model("math_symbols_model_0.9815_accuracy.h5")

while True:

    _, unprocessed_frame = cap.read()
    
    frame = cv2.cvtColor(unprocessed_frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    _, mask = cv2.threshold(frame, 80, 255, cv2.THRESH_BINARY_INV)

    symbols = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    for contour in contours:
        area = cv2.contourArea(contour)
        if 30 < area < 5000:
            x, y, w, h = cv2.boundingRect(contour)
            symbols.append(Symbol(unprocessed_frame, x, y, w, h, model))

    for symbol in symbols:
        symbol.draw_rect(unprocessed_frame)
        #symbol.show_img()

    #cv2.drawContours(unprocessed_frame, contours, -1, (0,255,0), 3)


    cv2.imshow("unprocessed", unprocessed_frame)
    cv2.imshow("processed", mask)

    if cv2.waitKey(1) == ord('q'):
        break