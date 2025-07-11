import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Dense, Flatten, Reshape, Dropout, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.saving import load_model
from tensorflow.keras import regularizers

from sklearn.model_selection import train_test_split
from PIL import Image
from numpy import array, expand_dims, prod

from sys import path
from pathlib import Path

# Конфигурация
IMAGE_DIR = Path(r"C:\My\Projects\images\main\Data_img\Dataset")
TARGET_HEIGHT = 320
TARGET_WIDTH = 320
SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# Проверка директории
if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Директория {IMAGE_DIR} не найдена")


def load_and_preprocess(path):
    img = Image.open(path).convert("L")
    img = img.resize((TARGET_HEIGHT, TARGET_WIDTH))
    img_array = array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = expand_dims(img_array, axis=-1)
    return img_array



input_shape = (TARGET_HEIGHT, TARGET_WIDTH, 1)

# def build_deep_autoencoder(img_shape, code_size):
#     # H,W,C = img_shape
#     # encoder
#     encoder = tf.keras.models.Sequential() # инициализация модели
#     encoder.add(Input(img_shape)) # добавление входного слоя, размер равен размеру изображения
#     encoder.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)))
#     encoder.add(MaxPooling2D(pool_size=2))
#     encoder.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)))
#     encoder.add(MaxPooling2D(pool_size=2))
#     encoder.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)))
#     encoder.add(MaxPooling2D(pool_size=2))
#     encoder.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)))
#     encoder.add(MaxPooling2D(pool_size=2))
#     encoder.add(Flatten())
#     encoder.add(Dropout(0.3))
#     encoder.add(Dense(code_size, name="getting_features"))
# 
#     # decoder
#     tenzor_size = int(img_shape[0] / 16)
#     decoder = tf.keras.models.Sequential()
#     decoder.add(Input((code_size,)))
#     decoder.add(Dense(tenzor_size*tenzor_size*256))
#     decoder.add(Reshape((tenzor_size, tenzor_size, 256)))
#     decoder.add(Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)))
#     decoder.add(Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)))
#     decoder.add(Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)))
#     decoder.add(Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=2, activation="sigmoid", padding='same', kernel_regularizer=regularizers.l2(1e-4)))
#     
#     return encoder, decoder
# 
# 
# encoder, decoder = build_deep_autoencoder(input_shape, code_size=128)
# encoder.summary()
# decoder.summary()
# 
# 
# inp = Input(input_shape)
# code = encoder(inp)
# reconstruction = decoder(code)
# 
# autoencoder_model = tf.keras.models.Model(inputs=inp, outputs=reconstruction)
# 
# 
# autoencoder_model.compile(
#         optimizer=Adamax(learning_rate=0.0001), 
#         loss=MeanSquaredError(),
#         metrics=["mae"]
#     )


for i in range(1, 19):
    print(f"\ntrain{i}\n")
    callback = EarlyStopping(
            monitor="val_loss", 
            patience=4,
            verbose=2,
            restore_best_weights=True
        )
    if i > 0: 
        autoencoder_model = load_model(Path(path[0] + f"/autoencoder_{i-1}.keras"))
        autoencoder_model.summary()
        
    train_paths = [p for p in (IMAGE_DIR / f"train{i}").glob('*') if p.suffix.lower() in SUPPORTED_EXT]
    # test_paths = [p for p in (IMAGE_DIR / "test").glob('*') if p.suffix.lower() in SUPPORTED_EXT]

    train_data = [load_and_preprocess(p) for p in train_paths]
    # test_data = [load_and_preprocess(p) for p in test_paths]

    train_data = array(train_data)
    # test_data = np.array(test_data)

    print(f"\nФорма обучающих данных: {train_data.shape}")  # Должно быть (n, 360, 360)
    # print(f"Форма тестовых данных: {test_data.shape}")


    history = autoencoder_model.fit(
            train_data,
            train_data,
            verbose="auto",
            epochs=20,
            batch_size=32,
            shuffle=True,
            validation_split=0.15,
            callbacks=[callback]
        )

    file_model = Path(path[0] + f"/autoencoder_{i}.keras")
    autoencoder_model.save(file_model)
