import cv2
import numpy as np

#symbol class which represents numbers, decimals, and operations
class Symbol():
    #initialize with dimensions of bounding box
    def __init__(self, x, y, w, h, webcam_img):
        webcam_img_height = webcam_img.shape[0]
        webcam_img_width = webcam_img.shape[1]
        PADDING = 4

        #adds a small padding of 4 pixels to each bounding box
        self.x = x - PADDING
        self.y = y - PADDING
        self.w = w + PADDING * 2
        self.h = h + PADDING * 2

        if self.x < 0:
            self.x = 0
        if self.y < 0:
            self.y = 0
        if self.x + self.w > webcam_img_width:
            self.w = webcam_img_width - self.x
        if self.y + self.h > webcam_img_height:
            self.h = webcam_img_height - self.y
        
        self.classification = None

    #processes an image of the symbol from the webcam image and classifies the symbol using a machine learning model
    def classify_symbol(self, model, classification_dict, webcam_img):

        #crop boundaries
        left_bound = self.x
        right_bound = self.x + self.w
        top_bound = self.y
        bottom_bound = self.y + self.h

        #crop image
        img_cropped = webcam_img[top_bound:bottom_bound, left_bound:right_bound]
        img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        _, img_processed = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY)

        #creates a padding of white pixels for the larger dimension that is 1/4 of it
        #pads the lesser dimension white such that the resulting image is squared
        if self.h > self.w:
            padding_top = self.h//4
            padding_side = (self.h + padding_top * 2 - self.w)//2
        else:
            padding_side = self.w//4
            padding_top = (self.w + padding_side * 2 - self.h)//2

        #apppends padding on top and bottom of the image
        padding = np.array([[255]*self.w]*padding_top, dtype=np.float32)
        img_padded = np.append(padding, img_processed, axis=0)
        img_padded = np.append(img_padded, padding, axis=0)
        
        #apppends padding on sides of the image
        padding = np.array([[255]*padding_side]*(self.h + padding_top * 2), dtype=np.float32)
        img_padded = np.append(padding, img_padded, axis=1)
        img_padded = np.append(img_padded, padding, axis=1)

        #preprocessing the image to input to the model
        img_processed = cv2.resize(img_padded, (100,100))
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
