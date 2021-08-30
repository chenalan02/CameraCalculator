from symbol import Symbol
import time
import cv2
import numpy as np
import copy


class Calculator:
    def __init__(self, model, categories, time_to_focus):
        self.time_to_focus = time_to_focus
        self.model = model
        self.CATEGORIES = categories
        self.max_displacement = 35

        self.symbols = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        self.time_num_symbols_changed = 0.0
        self.calculate = False
        self.answer = ""
        
        self.cap = cv2.VideoCapture(0)
        _, frame = self.cap.read()
        frame_shape = frame.shape
        height = frame_shape[0]
        width = frame_shape[1]
        self.unprocessed_frame = np.zeros((height, width))

    def update_bounding_boxes(self):

        new_symbols = self.process_bounding_boxes()

        if (len(new_symbols) != len(self.symbols)):
            self.time_num_symbols_changed = time.time()
            self.calculate = False
            self.answer = ""

        elif (time.time() - self.time_num_symbols_changed > self.time_to_focus and self.calculate == False):
            self.calculate = True
            self.classify_symbols()
            try:
                self.calculate_equation()
            except:
                self.time_num_symbols_changed = time.time()
                self.calculate = False
                self.answer = ""
        
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
            if 15 < area < 5000:
                x, y, w, h = cv2.boundingRect(contour)
                new_symbols.append(Symbol(x, y, w, h))
        
        return new_symbols
        
    def classify_symbols(self):
        for symbol in self.symbols:
            symbol.classify_symbol(self.model, self.CATEGORIES, self.unprocessed_frame)
        
    def calculate_equation(self):
        self.symbols.sort(key = lambda symbol:symbol.x, reverse= True)
        symbols_stack = []
        for symbol in self.symbols:
            symbols_stack.append(symbol.classification)
        
        operations_stack = []
        number = ""
        while (len(symbols_stack) != 0):
            digit = symbols_stack.pop()
            if (digit not in ['add','div','mul','sub']):
                number += digit
            else:
                operations_stack.insert(0, number)
                operations_stack.insert(0, digit)
                number = ""
        operations_stack.insert(0, number)

        while (len(operations_stack) > 1):
            num1 = float(operations_stack.pop())
            operation = operations_stack.pop()
            num2 = float(operations_stack.pop())

            if (operation == 'add'):
                result = num1 + num2
            elif (operation == 'div'):
                result = num1 / num2
            elif (operation == 'mul'):
                result = num1 * num2
            else:
                result = num1 - num2

            operations_stack.append(result)

        self.answer = operations_stack[0]
        
    def draw(self):
        for symbol in self.symbols:
            symbol.draw_rect(self.unprocessed_frame)
            symbol.draw_classification(self.unprocessed_frame)
        if (self.answer != ""):
            cv2.putText(self.unprocessed_frame, str(self.answer), (0,50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        cv2.imshow("webcam", self.unprocessed_frame)