import cv2
import numpy as np

class Symbol():
    def __init__(self, img, x, y, w, h, model):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        img_height = img.shape[0]
        img_width = img.shape[1]

        buffer = h//3
        left_bound = x - buffer
        if (left_bound < 0):
            left_bound = 0

        right_bound = x + w + buffer
        if (right_bound >= img_width):
            right_bound = img_width - 1

        top_bound = y - buffer
        if (top_bound < 0):
            top_bound = 0

        bottom_bound = y + h + buffer
        if (bottom_bound > img_height):
            bottom_bound = img_height - 1

        self.img = img[top_bound:bottom_bound, left_bound:right_bound]
        img_processed = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, img_processed = cv2.threshold(img_processed, 80, 255, cv2.THRESH_BINARY)
        img_processed = cv2.resize(img_processed, (100,100))
        img_processed = np.expand_dims(img_processed, axis = 2)
        img_processed = np.reshape(img_processed, (1, 100, 100, 1))
        img_processed = img_processed/255
        #self.classification = int(model.predict(img_processed).argmax(axis = 1))

    def show_img(self):
        cv2.imshow("cog", self.img)

    def draw_rect(self, img):
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 2)
