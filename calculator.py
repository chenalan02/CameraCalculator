from symbol import Symbol
import time
import cv2
import numpy as np


class Calculator:
    def __init__(self, model, categories, time_to_focus):
        self.symbols = []
        self.num_symbols = 0
        self.max_displacement = 25
        self.time_since_symbols_change = 0.0
        self.time_to_focus = time_to_focus
        self.calculate = False
        self.answer = ""
        self.model = model
        self.CATEGORIES = categories

        self.cap = cv2.VideoCapture(0)
        _, frame = self.cap.read()
        frame_shape = frame.shape
        self.height = frame_shape[0]
        self.width = frame_shape[1]
        self.unprocessed_frame = np.zeros((self.height, self.width))

    def update_bounding_boxes(self):

        new_symbols = self.process_bounding_boxes()

        if (len(new_symbols) != len(self.symbols)):
            self.time_num_symbols_changed = time.time()
            self.calculate = False

        elif (time.time() - self.time_num_symbols_changed > self.time_to_focus and self.calculate == False):
            self.calculate = True
            self.classify_symbols()
        
        if (self.calculate):
            for new_symbol in new_symbols:
                for old_symbol in self.symbols:
                    if (self.is_same_symbol(old_symbol, new_symbol)):
                        new_symbol.classification = old_symbol.classification
                        break
        self.symbols = new_symbols                
            
    def is_same_symbol(self, symbol1, symbol2):
        if (abs(symbol1.x - symbol2.x) < self.max_displacement and abs(symbol1.y - symbol2.y) < self.max_displacement):
            return True

    def process_bounding_boxes(self):

        _, self.unprocessed_frame = self.cap.read()
        frame = cv2.cvtColor(self.unprocessed_frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (15, 15), 0)
        _, mask = cv2.threshold(frame, 80, 255, cv2.THRESH_BINARY_INV)

        new_symbols = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        for contour in contours:
            area = cv2.contourArea(contour)
            if 30 < area < 5000:
                x, y, w, h = cv2.boundingRect(contour)
                new_symbols.append(Symbol(x, y, w, h))
        
        return new_symbols
        
    def classify_symbols(self):
        for symbol in self.symbols:
            symbol.classify_symbol(self.model, self.CATEGORIES, self.unprocessed_frame)
        

    def draw(self):
        for symbol in self.symbols:
            symbol.draw_rect(self.unprocessed_frame)
            symbol.draw_classification(self.unprocessed_frame)
        cv2.imshow("webcam", self.unprocessed_frame)