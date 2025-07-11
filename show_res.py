import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from PIL import Image
from pathlib import Path

# Функция для загрузки и предобработки изображения
def load_and_preprocess_image(path, target_size=(28, 28)):
    img = Image.open(path).convert('L')  # конвертируем в grayscale
    img = img.resize(target_size)
    img_array = np.array(img).astype('float32') / 255.0  # нормализация в [0,1]
    img_array = np.expand_dims(img_array, axis=(0, -1))  # (1, H, W, 1)
    return img_array

IMAGE_DIR = Path("C:\\My\\Projects\\images\\main\\Data_img\\Dataset\\test")
file_model = IMAGE_DIR / "g_ (1)_rotated_0.jpg"

# Определение простого автоэнкодера
latent_dim = 128
input_shape = (320, 320, 1)

auto = load_model("my_model_0.keras")

encoder = Model(inputs=auto.input, outputs=auto.get_layer("latent_features").output)
autoencoder = Model(inputs=auto.input, outputs=auto.get_layer(decoded).output)
encoder.summary()
autoencoder.summary()


# Загрузка и подготовка изображения
img_path = file_model  # замените на путь к вашему изображению
img = load_and_preprocess_image(img_path)

# Обучение автоэнкодера на одном изображении (для демонстрации)
# autoencoder.fit(img, img, epochs=100, verbose=0)

# Получение признаков и восстановления
features = encoder.predict(img)
reconstructed = autoencoder.predict(img)

# Визуализация
plt.figure(figsize=(12,4))

# Исходное изображение
plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(img[0,:,:,0], cmap='gray')
plt.axis('off')

# Визуализация признаков (latent_dim вектор)
plt.subplot(1,3,2)
plt.title("Encoded Features")
plt.bar(range(latent_dim), features[0])
plt.xlabel("Feature index")
plt.ylabel("Feature value")

# Восстановленное изображение
plt.subplot(1,3,3)
plt.title("Reconstructed Image")
plt.imshow(reconstructed[0,:,:,0], cmap='gray')
plt.axis('off')

plt.show()
