from symbol import Symbol
import time
import cv2
import numpy as np

class Calculator:
    """
    The Calculator class uses a camera to perform calculations of basic handwritten math equations written on paper
    Once the number of detected symbols on screen doesn't change for a specified number of seconds, a calculation is performed.
    The answer, symbol classifications, and bounding boxes stays on screen and follows the symbols
    """

    def __init__(self, model, categories, time_to_focus):
        """
        Initializes camera and attributes

        Parameters:
        ------------
        model: tensorflow model
            - model used to classify symbols
        categories: dict
            - dictionary mapping for symbols
        time_to_focus
            - time the calculator waits once the number of symbols stay constant before calculating the equation
        """
        self.model = model
        self.CATEGORIES = categories

        # time the calculator waits once the number of symbols stay constant before calculating the equation
        self.time_to_focus = time_to_focus
        # maximum pixels a symbol can move between consecutive frames to lose track of it
        self.max_displacement = 25

        # list of all symbols (numbers, decimals, operations)
        # set to large initial length so the calculator doesnt try to perform a calculation on the first frame in update_symbols()
        self.symbols = [0]*100

        # the last time when the number of symbols changed
        self.time_num_symbols_changed = 0.0

        # variable for whether the calculator has performed a calculation
        self.calculated = False

        # answer of the calculation
        self.answer = ""
        
        # captures a frame to initialize a black frame of the same dimensions
        self.cap = cv2.VideoCapture(0)
        _, frame = self.cap.read()
        frame_shape = frame.shape
        height = frame_shape[0]
        width = frame_shape[1]
        self.unprocessed_frame = np.zeros((height, width))


    def update_symbols(self):
        """
        updates the symbols, determines whether a calculation should be made, and keeps track of symbols from previous frames
        """

        # retreives symbols from newest frame
        new_symbols = self.process_symbols()

        # if the numbers of symbols have changed, update the time of last change and set calculate to false
        if len(new_symbols) != len(self.symbols):
            self.time_num_symbols_changed = time.time()
            self.calculated = False
            self.answer = ""

        # if it has been a set number of seconds since the last change in symbols, classify them and calculate the equation if applicable
        elif time.time() - self.time_num_symbols_changed > self.time_to_focus and self.calculated == False:
            self.calculated = True
            self.classify_symbols()

            # if the symbols do not create a valid equation, update the time of last change and set calculate to false
            try:
                self.calculate_equation()
            except:
                self.time_num_symbols_changed = time.time()
                self.calculated = False
                self.answer = ""

        # if calculate is true, track the position of the symbols from the previous frame in terms of proximity to keep classifications
        if self.calculated:
            for new_symbol in new_symbols:
                for old_symbol in self.symbols:
                    if self.is_same_symbol(old_symbol, new_symbol):
                        new_symbol.classification = old_symbol.classification
                        break
        # update symbols
        self.symbols = new_symbols

    def is_same_symbol(self, symbol1, symbol2):
        """
        determines if 2 symbols from consecutive frames are the same if they are within a specified number of pixels apart
        """
        if abs(symbol1.x - symbol2.x) < self.max_displacement and abs(symbol1.y - symbol2.y) < self.max_displacement:
            return True

    def process_symbols(self):
        """
        processes the symbols of a frame

        Returns:
        ---------
        new_symbols: list of Symbol objects
        """
        # decrease effects of noise by applying a gaussian blur and mask
        _, self.unprocessed_frame = self.cap.read()
        frame = cv2.cvtColor(self.unprocessed_frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.GaussianBlur(frame, (15, 15), 0)
        _, mask = cv2.threshold(frame, 80, 255, cv2.THRESH_BINARY_INV)

        new_symbols = []
        # detect symbols by finding contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        for contour in contours:
            # add a symbol if its contour is within a specific area range
            area = cv2.contourArea(contour)
            if 20 < area < 5000:
                x, y, w, h = cv2.boundingRect(contour)
                new_symbols.append(Symbol(x, y, w, h, self.unprocessed_frame))

        return new_symbols

    def classify_symbols(self):
        """
        classify all symbols in self.symbols
        """
        for symbol in self.symbols:
            symbol.classify_symbol(self.model, self.CATEGORIES, self.unprocessed_frame)

    def calculate_equation(self):
        """
        creates and calculates the equation ignoring bedmas/pemdas rules
        first concatenates numbers with decimals and/or multiple digits using symbols_stack as a container
        then performs math operations using operations_stack as a container
        """
        # sort the symbols in terms of x position from right to left to use as a stack and cleans division symbols
        self.symbols.sort(key = lambda symbol:symbol.x, reverse= True)
        self.clean_division_symbols()
        symbols_stack = []
        for symbol in self.symbols:
            symbols_stack.append(symbol.classification)
        
        # receives symbols from symbols_stack, concatenating them to form a number until a operation is received
        # once an operation is received, add both number and the operation to operations_stack and reset the number
        operations_stack = []
        number = ""
        while (len(symbols_stack) != 0):
            digit = symbols_stack.pop()
            if digit not in ['add','div','mul','sub']:
                number += digit
            else:
                operations_stack.insert(0, number)
                operations_stack.insert(0, digit)
                number = ""
        # add the last number of the equation once the stack is empty
        operations_stack.insert(0, number)

        # operations_stack should now be in alternating order of numbers and operations
        # perform the first operation at the top of the stack and append the result to the stack
        # repeat until only a single number is left in the stack
        while (len(operations_stack) > 1):
            num1 = float(operations_stack.pop())
            operation = operations_stack.pop()
            num2 = float(operations_stack.pop())

            if operation == 'add':
                result = num1 + num2
            elif operation == 'div':
                result = num1 / num2
            elif operation == 'mul':
                result = num1 * num2
            else:
                result = num1 - num2

            operations_stack.append(result)

        self.answer = operations_stack[0]

    def clean_division_symbols(self):
        """
        cleans the divion symbols since they are usually detected as 3 separate symbols
        the middle part of the division symbol should be recognized as a subtraction symbol while the dots are recognized as decimals

        the subtraction symbol should be preceeded by the 2 symbols recognized as decimals, these are deleted from self.symbols
        the subtraction symbol is changed to a division
        """
        # self.symbols should be sorted in reverse order of x position
        for i, symbol in enumerate(self.symbols):
            if symbol.classification == 'sub':
                previous_symbols = [symbol.classification for symbol in self.symbols[i-2:i]]
                if previous_symbols == ['.', '.']:
                    self.symbols[i].classification = 'div'
                    del self.symbols[i-1]
                    del self.symbols[i-2]

    def draw(self):
        """
        draw the bounding boxes, and classification/answer if applicable, to the webcam image
        """
        for symbol in self.symbols:
            symbol.draw_bounding_box(self.unprocessed_frame)
            symbol.draw_classification(self.unprocessed_frame)
        if self.answer != "":
            cv2.putText(self.unprocessed_frame, str(self.answer), (0,50), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        cv2.imshow("webcam", self.unprocessed_frame)