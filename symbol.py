import cv2
import numpy as np

class Symbol():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.classification = None

    def classify_symbol(self, model, classification_dict, webcam_img):

        webcam_img_height = webcam_img.shape[0]
        webcam_img_width = webcam_img.shape[1]

        if (self.h > self.w):
            padding = self.h//3
        else:
            padding = self.w//3

        left_bound = self.x - padding
        right_bound = self.x + self.w + padding
        top_bound = self.y - padding
        bottom_bound = self.y + self.h + padding

        width = right_bound - left_bound
        height = bottom_bound - top_bound
        dim_difference = abs(width - height)

        if (width > height):
            top_bound -= dim_difference//2
            bottom_bound += dim_difference//2
        else:
            left_bound -= dim_difference//2
            right_bound += dim_difference//2

        if (left_bound < 0):
            left_bound = 0
        if (right_bound >= webcam_img_width):
            right_bound = webcam_img_width - 1
        if (top_bound < 0):
            top_bound = 0
        if (bottom_bound > webcam_img_height):
            bottom_bound = webcam_img_height - 1

        img = webcam_img[top_bound:bottom_bound, left_bound:right_bound]

        img_processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_processed = cv2.threshold(img_processed, 80, 255, cv2.THRESH_BINARY)
        img_processed = cv2.resize(img_processed, (100,100))
        img_processed = np.expand_dims(img_processed, axis = 2)
        img_processed = np.reshape(img_processed, (1, 100, 100, 1))
        img_processed = img_processed/255

        classification_idx = int(model.predict(img_processed).argmax(axis = 1))
        self.classification = classification_dict[classification_idx]

    def draw_rect(self, img):
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 2)

    def draw_classification(self, img):
        if (self.classification != None):
            cv2.putText(img, str(self.classification), (self.x, self.y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
