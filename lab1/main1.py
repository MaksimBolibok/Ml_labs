
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import GlorotNormal
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt

# Гіперпараметри
learning_rate = 0.05
nEpochs = 500
hidden_units = 4

# Вхідні дані для XOR
features = np.array([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1]
])
labels = np.array([
    [0, 1],
    [1, 0],
    [1, 0],
    [0, 1]
])

# Архітектура моделі
initializer = GlorotNormal(seed=12)
model = Sequential([
    Dense(units=hidden_units, input_shape=(2,), kernel_initializer=initializer, activation='sigmoid'),
    Dense(units=2, kernel_initializer=initializer, activation='softmax')
])
model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(learning_rate=learning_rate, momentum=0.95, nesterov=True)
)

# Навчання
print(f"Training started (learning rate = {learning_rate}, hidden units = {hidden_units})")
history = model.fit(features, labels, epochs=nEpochs, batch_size=1, verbose=0)
print("Training finished")

# Оцінка
print("Evaluation on training set:")
loss = model.evaluate(features, labels, verbose=0)
print(f"Final loss: {loss:.4f}")

# Прогноз
print("Prediction for [0, 0] (True XOR False):")
pred = model.predict(np.array([[0, 0]]), verbose=0)
print(pred)
print("Class:", np.argmax(pred))

# Побудова графіка
plt.title("Loss curve during training")
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
