import cv2
import numpy as np
from tensorflow.keras.models import load_model
from symbol import Symbol
from calculator import Calculator

CATEGORIES = {0: '0' , 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'add', 11: '.', 12: 'div', 13: 'mul', 14: 'sub'}

model = load_model("symbol_recognizer_model/math_symbols_model_0.9815_accuracy.h5")
black_img = np.zeros((1, 100, 100, 1))
pred = model.predict(black_img)

calculator = Calculator(model, CATEGORIES, 1.5)

while True:

    calculator.update_symbols()
    calculator.draw()

    if cv2.waitKey(1) in [ord('q'), ord('Q')]:
        break