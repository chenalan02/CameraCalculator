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

        img_height = webcam_img.shape[0]
        img_width = webcam_img.shape[1]

        padding = self.h//3
        left_bound = self.x - padding
        if (left_bound < 0):
            left_bound = 0

        right_bound = self.x + self.w + padding
        if (right_bound >= img_width):
            right_bound = img_width - 1

        top_bound = self.y - padding
        if (top_bound < 0):
            top_bound = 0

        bottom_bound = self.y + self.h + padding
        if (bottom_bound > img_height):
            bottom_bound = img_height - 1

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
