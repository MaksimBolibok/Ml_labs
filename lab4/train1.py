from keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras
import numpy as np

# Параметри
learning_rate = 0.001
epochs = 15
batch_size = 64
n_iny = 224
n_inx = 224

# Завантаження зображень
train_path = "train"
test_path = "test"
generator = ImageDataGenerator(preprocessing_function=keras.applications.vgg16.preprocess_input)

traingen = generator.flow_from_directory(train_path, target_size=(n_iny, n_inx), batch_size=batch_size)
testgen = generator.flow_from_directory(test_path, target_size=(n_iny, n_inx), batch_size=batch_size)

# Завантаження моделі
model_base = keras.applications.VGG16(weights='imagenet', input_shape=(n_iny, n_inx, 3), include_top=True)
model_base.trainable = False
model_base.summary()

# Нова модель
model = keras.Sequential()
model.add(model_base)
model.add(Dense(3, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(traingen, epochs=epochs, validation_data=testgen)

model.save("train1.keras")