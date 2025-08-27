import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, UpSampling2D, 
                                     Input, Dense, Flatten, Reshape, Dropout, 
                                     SpatialDropout2D, BatchNormalization, Activation)
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
IMAGE_DIR = Path(r"C:\My\Projects\images\main\Data_img\Dataset_192")
TARGET_HEIGHT = 192
TARGET_WIDTH = 192
SUPPORTED_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# Проверка директории
if not IMAGE_DIR.exists():
    raise FileNotFoundError(f"Директория {IMAGE_DIR} не найдена")


# @tf.function
# def combined_loss(y_true, y_pred):
#     mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
#     mae = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
#     return 0.5 * mse + 0.5 * mae

# # 🔧 Регистрируем loss, чтобы она сохранялась с моделью
# get_custom_objects().update({'combined_loss': combined_loss})

def load_and_preprocess(path):
    img = Image.open(path).convert("L")
    img = img.resize((TARGET_WIDTH, TARGET_HEIGHT))
    img_array = array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = expand_dims(img_array, axis=-1)
    return img_array



input_shape = (TARGET_HEIGHT, TARGET_WIDTH, 1)
inputs = Input(shape=input_shape)

# Encoder
x = Conv2D(64, kernel_size=3, activation="relu", padding='same', kernel_regularizer=regularizers.L2(1e-5))(inputs)
x = MaxPooling2D(pool_size=2, padding='same')(x)  

x = Conv2D(128, kernel_size=3, activation="relu", padding='same', kernel_regularizer=regularizers.L2(1e-5))(x)
x = MaxPooling2D(pool_size=2, padding='same')(x)  
                                                 
x = Conv2D(256, kernel_size=3, activation="relu", padding='same', kernel_regularizer=regularizers.L2(1e-5))(x)
x = MaxPooling2D(pool_size=2, padding='same')(x)  
                                                 
x = Conv2D(256, kernel_size=3, activation="relu", padding='same', kernel_regularizer=regularizers.L2(1e-5))(x)
x = MaxPooling2D(pool_size=2, padding='same')(x)  

shape_before_flattening = x.shape[1:]   # Сохраним форму для декодера
flattened = Flatten()(x)
flattened = Dropout(0.2)(flattened)
latent_features = Dense(512, activation=None, name='latent_features', kernel_regularizer=None)(flattened)

# Decoder
x = Dense(prod(shape_before_flattening), activation='relu', kernel_regularizer=None)(latent_features)
x = Reshape(shape_before_flattening)(x)

x = UpSampling2D(size=2)(x)                                                                      
x = Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.L2(1e-5))(x)

x = UpSampling2D(size=2)(x)                                                                     
x = Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.L2(1e-5))(x)

x = UpSampling2D(size=2)(x)                                                                     
x = Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.L2(1e-5))(x)

x = UpSampling2D(size=2)(x)    
x = Conv2D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.L2(1e-5))(x)

decoded = Conv2D(1, kernel_size=3, activation='sigmoid', padding='same', name="decoder")(x)


autoencoder_model = Model(inputs, decoded)
autoencoder_model.compile(
        optimizer=Adam(learning_rate=0.0005), 
        loss=MeanAbsoluteError(),
        metrics=["mse"]
    )
autoencoder_model.summary()


for i in range(0, 3):

    # autoencoder_model = load_model(Path(path[0] + f"/new_model_1.keras"))
    
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
        autoencoder_model = load_model(Path(path[0] + f"/new_model_{1}.keras"))
        autoencoder_model.summary()
        
    train_paths = [p for p in (IMAGE_DIR / f"train_{0}").glob('*') if p.suffix.lower() in SUPPORTED_EXT]
    # train_paths = [p for p in (IMAGE_DIR / f"image_10").glob('*') if p.suffix.lower() in SUPPORTED_EXT]
    train_data = [load_and_preprocess(p) for p in train_paths]
    train_data = array(train_data)
    
    # # Для обучения на одном изображении
    # train_paths = IMAGE_DIR / f"train_{i}/g_ (11)_rotated_180.jpg"
    # train_data = load_and_preprocess(train_paths)
    # train_data = expand_dims(array(train_data), axis=0)
    

    print(f"\nФорма обучающих данных: {train_data.shape}")  


    history = autoencoder_model.fit(
            train_data,
            train_data,
            verbose=2,
            epochs=epochs,
            batch_size=32,
            shuffle=True,
            validation_split=0.15,
            # callbacks=[callback]
        )

    file_model = Path(path[0] + f"/new_model_{1}.keras")
    autoencoder_model.save(file_model)
