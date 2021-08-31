# RealTimeCameraCalculator
A Camera that can calculate written math calculations in real time, including multiple operations and decimals. Once the number of detected symbols(numbers, operations, decimals) on screen doesnt change for a specified number of seconds, a calculation is performed. The answer, symbol classifications, and bounding boxes stays on screen and follows the symbols until the number of detected symbols change.

Uses openCV to process the webcam image and retreive bounding boxes for each symbol (numbers, operations, decimals). Uses a tensorflow trained CNN model to identify each symbol. The identifier model was trained using a kaggle dataset found at https://www.kaggle.com/sagyamthapa/handwritten-math-symbols. 

<p float="left">
  <img src="https://github.com/chenalan02/CameraCalculator/blob/main/Readme%20Images/Screenshot%202021-08-30%20212950.png" width = 480/>
  <img src="https://github.com/chenalan02/CameraCalculator/blob/main/Readme%20Images/Screenshot%202021-08-30%20180311.png" width = 480/> 
</p>

## Training the Identifier Model

The model was able to reach an accuracy of >90% accuracy within 5 epochs but slowly made improvements until it reached a high of 98% accuracy and a low in loss around epoch 30. Related files can be found in symbol_recognizer_model/


<p float="left">
  <img src="https://github.com/chenalan02/CameraCalculator/blob/main/Readme%20Images/acc.png" width = 480/>
  <img src="https://github.com/chenalan02/CameraCalculator/blob/main/Readme%20Images/loss.png" width = 480/> 
</p>

![Confusion Matrix](https://github.com/chenalan02/CameraCalculator/blob/main/Readme%20Images/confusion%20matrix.png)

The confusion matrix shows common incorrect identifications between numbers such as 2 and 3, 6 and 8, division and subtraction etc. During the implementation of the camera calculator these misidentifications occasionally occured.

## Camera Calculator Documentation
The Camera Calculator was implemented using objects to represent each symbol and a calculator class to handle the operation of the calculator as a whole.

### symbol.py

`symbol.Symbol(x, y, w, h)`
> The Symbol class represents a single symbol(number/operation/decimal) and its bounding box

`self.x` - x position of the top left corner of the bounding box\
`self.y` - y position of the top left corner of the bounding box\
`self.w` - width of the bounding box\
`self.h` - height of the bounding box\
`self.classification` -  height of the bounding box

`def classify_symbol(self, model, classification_dict, webcam_img)`
* method to process a trimmed image of the symbol from the webcam image to classify the symbol using a machine learning model
* `model` - CNN model used to identify the symbol
* `classification_dict` - dictionary mapping of the symbol names from their integer encoding
* `webcam_img` - unprocessed image from the webcam

`def draw_bounding_box(self, img)`
* method to draw the bounding box of the symbol
* `img` - image to draw the bounding box to

`def draw_classification(self, img)`
* method to draw the classification of the symbol in text
* `img` - image to draw the classification to

### calculator.py

`calculator.Symbol(model, categories, time_to_focus)`
> The Calculator class represents the calculator as a whole and manages the tracking of symbols and calculations

`self.model` - CNN model used to recognize symbols
`self.CATEGORIES` - dictionary mapping of the symbol names from their integer encoding
`self.time_to_focus` - number of seconds the calculator should wait once number of symbols stay constant before performing a calculation
`self.max_displacement` - maximum pixels a symbol can move between consecutive frames to lose track of it
`self.symbols` - list of all symbols
`self.time_num_symbols_changed` - the last time that the number of symbols changed or a calculation failed 
`self.calculate` - boolean for whether a calculation has been performed
`self.answer` - the answer of the calculation
`self.cap` - the camera used by the OpenCV Library to capture frames
`self.unprocessed_frame` - the latest unprocessed frame captured by the webcam

`def update_symbols(self)`
* updates the symbols, determines whether a calculation should be made, and keeps track of symbols from previous frames

`def is_same_symbol(self, symbol1, symbol2)`
* determines if 2 symbols from consecutive frames are the same if they are within a specified number of pixels apart

`def process_symbols(self)`
* captures a new frame and processes the bounding boxes and symbols

`def classify_symbols(self)`
* classifies all the symbols in `self.symbols`

`def calculate_equation(self)`
* calculates the equation ignoring bedmas/pemdas rules
* first concatenates numbers with decimals and/or multiple digits using a stack
* then performs math operations using another stack

`def clean_division_symbols(self)`
* cleans the division symbols since they are detected as 3 separate symbols

`def draw(self)`
* draw the bounding boxes, and classification/answer if applicable, to the webcam image
