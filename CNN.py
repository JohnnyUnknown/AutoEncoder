import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, UpSampling2D, 
                                     Input, BatchNormalization, LeakyReLU)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from tensorflow.image import ssim
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.saving import load_model, get_custom_objects
from tensorflow.keras import regularizers

from sklearn.model_selection import train_test_split
from PIL import Image
from numpy import array, expand_dims, prod

from sys import path
from pathlib import Path

# Конфигурация
IMAGE_DIR = Path(r"C:\My\Projects\images\main\Data_img\Dataset_256")
TARGET_HEIGHT = 256
TARGET_WIDTH = 256
LATENT_DIM = 128 
SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# Проверка директории
if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Директория {IMAGE_DIR} не найдена")


# Предобработка
def load_and_preprocess(path):
    img = Image.open(path).convert("L")
    img = img.resize((TARGET_WIDTH, TARGET_HEIGHT))
    img_array = array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = expand_dims(img_array, axis=-1)
    return img_array


# Вход
inputs = Input(shape=(TARGET_HEIGHT, TARGET_WIDTH, 1), name="input_image")

# === Encoder (извлекает признаки) ===
x = Conv2D(64, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(1e-4))(inputs)
# x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=2, padding='same')(x)  # 128x128

x = Conv2D(128, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
# x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=2, padding='same')(x)  # 64x64

x = Conv2D(128, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
# x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)
x = MaxPooling2D(pool_size=2, padding='same')(x)  # 32x32

# Латентная карта признаков — эту мы будем использовать для сравнения!
x = Conv2D(LATENT_DIM, kernel_size=3, activation='relu', padding='same', name='features')(x)  # 32x32x128

# === Decoder (только если нужно обучать как автоэнкодер) ===
x = UpSampling2D(size=2)(x)  # 64x64
x = Conv2D(128, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
# x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)

x = UpSampling2D(size=2)(x)  # 128x128
x = Conv2D(128, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
# x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)

x = UpSampling2D(size=2)(x)  # 256x256
x = Conv2D(64, kernel_size=3, padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
# x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.1)(x)

decoded = Conv2D(1, kernel_size=3, activation='sigmoid', padding='same', name='decoder_output')(x)

# Создаём модель
autoencoder = Model(inputs, decoded, name="geo_autoencoder")


autoencoder.compile(
        optimizer=Adam(learning_rate=0.0001), 
        loss=MeanSquaredError(),
        metrics=["mae"]
    )
autoencoder.summary()


for i in range(0, 1):

    # autoencoder = load_model(Path(path[0] + f"/CNN_{0}.keras"), compile=False)
    # autoencoder.compile(
    #     optimizer=Adam(learning_rate=0.00005),
    #     loss=MeanAbsoluteError(),
    #     metrics=["mse"]
    # )
    
    # epochs = (10 if not i else 20)
    epochs = 10
    print(f"\ntrain_{i}\n")
    callback = EarlyStopping(
            monitor="val_loss", 
            patience=5,
            verbose=2,
            restore_best_weights=True
        )
    if i > 0: 
        autoencoder = load_model(Path(path[0] + f"/CNN_{0}.keras"))
        autoencoder.summary()
        
    train_paths = [p for p in (IMAGE_DIR / f"train_{0}").glob('*') if p.suffix.lower() in SUPPORTED_EXT]
    # train_paths = [p for p in (IMAGE_DIR / f"image_10").glob('*') if p.suffix.lower() in SUPPORTED_EXT]
    train_data = [load_and_preprocess(p) for p in train_paths]
    train_data = array(train_data)
    
    # # Для обучения на одном изображении
    # train_paths = IMAGE_DIR / f"train_{i}/g_1-5_rot_90.jpg"
    # train_data = load_and_preprocess(train_paths)
    # train_data = expand_dims(array(train_data), axis=0)
    

    print(f"\nФорма обучающих данных: {train_data.shape}")  


    history = autoencoder.fit(
            train_data,
            train_data,
            verbose=2,
            epochs=epochs,
            batch_size=32,
            shuffle=True,
            validation_split=0.15,
            callbacks=[callback]
        )

    file_model = Path(path[0] + f"/CNN_{0}.keras")
    autoencoder.save(file_model)
