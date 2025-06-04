import keras
import cv2 as cv
import numpy as np

# Цим файлом я вирішив протестувати дообучені моделі
model = keras.models.load_model("train1.keras")  # або train2.keras

# Обробка
file_name = "car.jpg"  # назва картинки
img = cv.imread(file_name)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img1 = cv.resize(img, [224, 224])
img1 = keras.applications.vgg16.preprocess_input(img1)
img1 = np.expand_dims(img1, axis=0)

# Який клас (0, 1 или 2)
res = model.predict(img1)
print("Клас:", np.argmax(res))