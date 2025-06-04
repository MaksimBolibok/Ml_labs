
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import GlorotNormal
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt

# Гіперпараметри
learning_rate = 0.1
nEpochs = 300
hidden_units = 4

# Вхідні дані
inputs_xyz = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
])


outputs_or = np.array([[0, 1] if (x or y or z) else [1, 0] for x, y, z in inputs_xyz])

# Архітектура нейромережі
initializer = GlorotNormal(seed=42)
model = Sequential([
    Dense(units=hidden_units, input_shape=(3,), kernel_initializer=initializer, activation='sigmoid'),
    Dense(units=2, kernel_initializer=initializer, activation='softmax')
])
model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
)

# Навчання
print("Training model for x OR y OR z")
history = model.fit(inputs_xyz, outputs_or, epochs=nEpochs, batch_size=1, verbose=0)
print("Training complete")

# Оцінка
print("Evaluation on full OR dataset:")
loss = model.evaluate(inputs_xyz, outputs_or, verbose=0)
print(f"Loss: {loss:.4f}")

# Перевірка передбачення
print("Predicting for input [0, 0, 0]:")
predict_result = model.predict(np.array([[0, 0, 0]]), verbose=0)
print(predict_result)
print("Class:", np.argmax(predict_result))

# Побудова графіка
plt.title("Loss during training (OR logic)")
plt.plot(history.history['loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()
