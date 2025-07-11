import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input, Dense, Flatten, Reshape, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
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
inputs = Input(shape=input_shape)

# Encoder
x = Conv2D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(inputs)
x = Conv2D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(inputs)
x = MaxPooling2D(pool_size=2, padding='same')(x)  
x = Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
x = Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
x = MaxPooling2D(pool_size=2, padding='same')(x)                                                   
x = Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
x = Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
x = MaxPooling2D(pool_size=2, padding='same')(x)                                                   
x = Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
x = Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
x = MaxPooling2D(pool_size=2, padding='same')(x)                                                   
x = Conv2D(512, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
# x = Conv2D(512, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
x = MaxPooling2D(pool_size=2, padding='same')(x)  

# Сохраним форму для декодера
shape_before_flattening = x.shape[1:]  

flattened = Flatten()(x)
flattened = Dropout(0.3)(flattened)
latent_features = Dense(128, activation='relu', name='latent_features', kernel_regularizer=regularizers.l1(1e-4))(flattened)

# Decoder
x = Dense(prod(shape_before_flattening), activation='relu', kernel_regularizer=regularizers.l1(1e-4))(latent_features)
x = Reshape(shape_before_flattening)(x)
# x = Conv2D(512, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
x = Conv2D(512, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
x = UpSampling2D(size=2)(x)                                                                        
x = Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
x = Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
x = UpSampling2D(size=2)(x)                                                                        
x = Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
x = Conv2D(256, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
x = UpSampling2D(size=2)(x)                                                                        
x = Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
x = Conv2D(128, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
x = UpSampling2D(size=2)(x)    
x = Conv2D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
x = Conv2D(64, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x)
x = UpSampling2D(size=2)(x) 
decoded = Conv2D(1, kernel_size=3, activation='sigmoid', padding='same')(x)



autoencoder_model = Model(inputs, decoded)
autoencoder_model.compile(
        optimizer=Adam(learning_rate=0.0001), 
        loss=MeanSquaredError(),
        metrics=["mae"]
    )
autoencoder_model.summary()


for i in range(0, 19):
    epochs = (60 if not i else 30)
    print(f"\ntrain{i}\n")
    callback = EarlyStopping(
            monitor="val_loss", 
            patience=4,
            verbose=2,
            restore_best_weights=True
        )
    if i > 0: 
        autoencoder_model = load_model(Path(path[0] + f"/my_model_{i-1}.keras"))
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
            epochs=epochs,
            batch_size=32,
            shuffle=True,
            validation_split=0.15,
            callbacks=[callback]
        )

    file_model = Path(path[0] + f"/my_model_{i}.keras")
    autoencoder_model.save(file_model)
