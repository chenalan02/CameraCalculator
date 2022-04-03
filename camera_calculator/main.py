import cv2
import numpy as np
from tensorflow.keras.models import load_model
from calculator import Calculator

# symbol mapping for model
CATEGORIES = {0: '0' , 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'add', 11: '.', 12: 'div', 13: 'mul', 14: 'sub'}

# load model and make prediction to initialize nvidea cuda files (lags the program for a few seconds during first model prediciton)
model = load_model("symbol_recognizer_model/math_symbols_model_0.9815_accuracy.h5")
black_img = np.zeros((1, 100, 100, 1))
pred = model.predict(black_img)

# initialize calculator with symbol mapping and time to focus before a calculation
time_to_focus = 1.5
calculator = Calculator(model, CATEGORIES, time_to_focus)

while True:

    calculator.update_symbols()
    calculator.draw()

    # program ends when the q key is pressed
    if cv2.waitKey(1) in [ord('q'), ord('Q')]:
        break