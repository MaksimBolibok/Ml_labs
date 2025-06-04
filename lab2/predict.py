
import numpy as np
from keras.models import load_model

# Завантаження точки
point = np.array([[0.0, 0.0]])

# Завантаження моделі
model = load_model("multi_model.keras")

# Прогноз
prediction = model.predict(point)
print("Ймовірності:", prediction)
print("Клас:", np.argmax(prediction))
