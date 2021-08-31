import cv2
import numpy as np

#symbol class which represents numbers, decimals, and operations
class Symbol():
    #initialize with dimensions of bounding box
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.classification = None

    #method to process a image of the symbol from the webcam image to classify the symbol using a machine learning model
    def classify_symbol(self, model, classification_dict, webcam_img):
        #detemine dimensions of webcam image
        webcam_img_height = webcam_img.shape[0]
        webcam_img_width = webcam_img.shape[1]

        #create a padding that is a quarter of the largest dimension of the bounding box
        if (self.h > self.w):
            padding = self.h//4
        else:
            padding = self.w//4

        #apply padding to each side
        left_bound = self.x - padding
        right_bound = self.x + self.w + padding
        top_bound = self.y - padding
        bottom_bound = self.y + self.h + padding

        #determine the difference between width and height
        width = right_bound - left_bound
        height = bottom_bound - top_bound
        dim_difference = abs(width - height)

        #use the dimensional difference to square the image (expand smaller dimensions to equal the larger one)
        if (width > height):
            top_bound -= dim_difference//2
            bottom_bound += dim_difference//2
        else:
            left_bound -= dim_difference//2
            right_bound += dim_difference//2

        #fixes processed bounds if they exceed the range of the webcam image
        if (left_bound < 0):
            left_bound = 0
        if (right_bound >= webcam_img_width):
            right_bound = webcam_img_width - 1
        if (top_bound < 0):
            top_bound = 0
        if (bottom_bound > webcam_img_height):
            bottom_bound = webcam_img_height - 1

        #take image of symbol from webcam image using bounds
        img = webcam_img[top_bound:bottom_bound, left_bound:right_bound]

        #preprocessing the image to input to the model
        img_processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img_processed = cv2.threshold(img_processed, 80, 255, cv2.THRESH_BINARY)
        img_processed = cv2.resize(img_processed, (100,100))
        img_processed = np.expand_dims(img_processed, axis = 2)
        img_processed = np.reshape(img_processed, (1, 100, 100, 1))
        img_processed = img_processed/255

        #make prediction
        classification_idx = int(model.predict(img_processed).argmax(axis = 1))
        self.classification = classification_dict[classification_idx]

    #draw the bounding box
    def draw_bounding_box(self, img):
        cv2.rectangle(img, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 2)

    #draw the classification
    def draw_classification(self, img):
        if (self.classification != None):
            cv2.putText(img, str(self.classification), (self.x, self.y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 2)
