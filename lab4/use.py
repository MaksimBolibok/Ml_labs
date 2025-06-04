import keras
import cv2 as cv
import numpy as np
from keras.applications.vgg16 import decode_predictions

# Завантаження моделі VGG16
model = keras.applications.VGG16(weights='imagenet')

# Обробка зображення
file_name = "car.jpg"  # назва картинки
img = cv.imread(file_name)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img1 = cv.resize(img, [224, 224])
img1 = keras.applications.vgg16.preprocess_input(img1)
img1 = np.expand_dims(img1, axis=0)  # або reshape(1, 224, 224, 3)

# Класифікація
res = model.predict(img1)
print(decode_predictions(res, top=4))

model.summary()