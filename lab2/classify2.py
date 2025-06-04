
import keras
import numpy as np
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd

# створення набору даних, зчитаних з csv файлу
def csv_reader(file_name):
    csv_dataset = pd.read_csv(
        file_name,
        names=["Label", "x", "y"])
    dataset = csv_dataset.copy()
    dataset_labels = keras.utils.to_categorical(
        dataset.pop('Label'))  # перетворює у one-hot-encoding подання, у бінарний вектор для кожного значення
    dataset_features = np.array(dataset)
    return dataset_features, dataset_labels

learning_rate = 0.05
nEpochs = 10
batch_size = 4

# Завантаження даних
features_train, labels_train = csv_reader("saturn_data_train.csv")
features_val, labels_val = csv_reader("saturn_data_eval.csv")

# Архітектура моделі
initializer = keras.initializers.GlorotNormal(seed=42)
model = keras.Sequential([
    Dense(units=4, input_shape=(2,), kernel_initializer=initializer, activation='relu'),
    Dense(units=2, kernel_initializer=initializer, activation='softmax')
])
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
)

# Навчання моделі
print("Початок навчання моделі SaturnClassifier")
history = model.fit(features_train, labels_train, epochs=nEpochs, batch_size=batch_size, verbose=1)
print("Навчання завершено")

# Оцінка моделі
print("Оцінка моделі на тестовому наборі:")
model.evaluate(features_val, labels_val)

# Передбачення для точки
test_input = np.array([[0, 0]])
prediction = model.predict(test_input)
print("Ймовірності:", prediction)
print("Прогнозований клас:", np.argmax(prediction))

# Побудова графіка втрат
plt.title("Зміна втрат протягом навчання")
plt.plot(history.history['loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()
