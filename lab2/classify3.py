
import keras
import numpy as np
import pandas as pd
from keras.layers import Dense
import matplotlib.pyplot as plt

# Завантаження CSV
def load_data(path):
    df = pd.read_csv(path, names=["Label", "x", "y"])
    labels = keras.utils.to_categorical(df.pop("Label"))
    features = np.array(df)
    return features, labels

# Зчитування даних
X_train, y_train = load_data("train_multiclass.csv")
X_test, y_test = load_data("test_multiclass.csv")

# Конфігурація моделі
model = keras.Sequential([
    Dense(6, input_shape=(2,), activation="relu", kernel_initializer="glorot_uniform"),
    Dense(3, activation="softmax")
])
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    metrics=["accuracy"]
)

# Навчання
print("Training multi-class classifier...")
history = model.fit(X_train, y_train, epochs=200, batch_size=8, verbose=0)
print("Done.")

# Оцінка
print("Evaluating on test set:")
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss: {loss:.4f} | Accuracy: {accuracy:.4f}")

# Передбачення
sample = np.array([[0, 0]])
prediction = model.predict(sample, verbose=0)
print("Prediction for [0, 0]:", prediction)
print("Predicted class:", np.argmax(prediction))

# Побудова графіка втрат
plt.title("Training Loss (Multi-class)")
plt.plot(history.history['loss'])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()

# Збереження моделі
model.save("multi_model.keras")
