import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.saving import get_custom_objects
from PIL import Image
from pathlib import Path


def load_and_preprocess_image(path, target_size):
    img = Image.open(path).convert('L')  
    img = img.resize(target_size)
    img_array = np.array(img).astype('float32') / 255.0  
    img_array = np.expand_dims(img_array, axis=(0, -1))  # (1, H, W, 1)
    return img_array

# def combined_loss(y_true, y_pred):
#     mse = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
#     mae = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
#     return 0.5 * mse + 0.5 * mae



# IMAGE_DIR = Path("C:\\My\\Projects\\images\\main\\Data_img\\Dataset_192\\train_0")
# file_model = "C:\\My\\Projects\\images\\main\\Data_img\\Dataset_192\\test\\g_ (10)_rotated_90_flipped_channels_permuted.jpg"
# file_model = IMAGE_DIR / "g_ (8)_rotated_270_flipped.jpg"
# file_model = IMAGE_DIR / "g_ (6)_rotated_0.jpg"


IMAGE_DIR = Path("C:\\My\\Projects\\images\\main\\Data_img\\Dataset_160\\train_0")
# file_model = "C:\\My\\Projects\\images\\main\\Data_img\\Dataset_160\\test\\g_ (9)_rotated_90.jpg"
file_model = IMAGE_DIR / "g_ (6)_rotated_0.jpg"
# file_model = IMAGE_DIR / "g_ (7)_rotated_270_flipped_channels_permuted.jpg"

# Определение простого автоэнкодера
latent_dim = 256
# input_shape = (192, 192, 1)
input_shape = (160, 160, 1)

# get_custom_objects().update({'combined_loss': combined_loss})
# autoencoder = load_model("new_model_1.keras")
autoencoder = load_model("model_256_160p_0.keras")

# encoder = Model(inputs=auto.input, outputs=auto.get_layer("getting_features").output)
encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer("latent_features").output)
# autoencoder = Model(inputs=auto.input, outputs=auto.get_layer("sequential_1").output)
# autoencoder = Model(inputs=auto.input, outputs=auto.get_layer("decoder").output)
autoencoder.summary()


# Загрузка и подготовка изображения
img = load_and_preprocess_image(file_model, input_shape[:2])

# Получение признаков и восстановления
features = encoder.predict(img)
reconstructed = autoencoder.predict(img)

# Визуализация
fig = plt.figure(figsize=(12,4))
axs = fig.subplots(1, 3)

axs[0].imshow(img[0,:,:,0], cmap='gray')
axs[0].axis('off')
axs[1].bar(range(latent_dim), features[0])
axs[2].imshow(reconstructed[0,:,:,0], cmap='gray')
axs[2].axis('off')

plt.show()


# metrics = auto.evaluate(img, reconstructed)
# print("\nМетрики my_model:", metrics, "\n\n")