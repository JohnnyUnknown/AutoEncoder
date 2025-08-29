import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.saving import get_custom_objects
from PIL import Image
from pathlib import Path


target_size = 256
latent_dim = 256
input_shape = (target_size, target_size, 1)

def load_and_preprocess_image(path, target_size):
    img = Image.open(path).convert('L')  
    img = img.resize(target_size)
    img_array = np.array(img).astype('float32') / 255.0  
    img_array = np.expand_dims(img_array, axis=(0, -1))  # (1, H, W, 1)
    return img_array


# 160x160
# IMAGE_DIR = Path(f"C:\\My\\Projects\\images\\main\\Data_img\\Dataset_{target_size}\\train_0")
file_model_1 = f"C:\\My\\Projects\\images\\main\\Data_img\\Dataset_{target_size}\\train_0\\g_1-5_rot_90.jpg"
file_model_2 = f"C:\\My\\Projects\\images\\main\\Data_img\\Dataset_{target_size}\\train_0\\g_1-5_rot_90.jpg"
file_model_3 = f"C:\\My\\Projects\\images\\main\\Data_img\\Dataset_{target_size}\\train_0\\g_1-5_rot_90.jpg"



autoencoder = load_model("CNN_0.keras")


encoder = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer("features").output)
autoencoder.summary()


img_1 = load_and_preprocess_image(file_model_1, input_shape[:2])
img_2 = load_and_preprocess_image(file_model_2, input_shape[:2])
img_3 = load_and_preprocess_image(file_model_3, input_shape[:2])

features_1 = encoder.predict(img_1)
reconstructed_1 = autoencoder.predict(img_1)
features_2 = encoder.predict(img_2)
reconstructed_2 = autoencoder.predict(img_2)
features_3 = encoder.predict(img_3)
reconstructed_3 = autoencoder.predict(img_3)

# Визуализация
fig = plt.figure(figsize=(12,8))
axs = fig.subplots(2, 3)

axs[0][0].imshow(img_1[0,:,:,0], cmap='gray')
axs[0][0].axis('off')
axs[1][0].imshow(reconstructed_1[0,:,:,0], cmap='gray')
axs[1][0].axis("off")

axs[0][1].imshow(img_2[0,:,:,0], cmap='gray')
axs[0][1].axis('off')
axs[1][1].imshow(reconstructed_2[0,:,:,0], cmap='gray')
axs[1][1].axis("off")

axs[0][2].imshow(img_3[0,:,:,0], cmap='gray')
axs[0][2].axis('off')
axs[1][2].imshow(reconstructed_3[0,:,:,0], cmap='gray')
axs[1][2].axis("off")

axs[0][0].set_title('Тестовое изображение')
axs[0][1].set_title('Тренировочное с 1 этапа')
axs[0][2].set_title('Тренировочное с 2 этапа')

plt.show()


# metrics = auto.evaluate(img, reconstructed)
# print("\nМетрики my_model:", metrics, "\n\n")