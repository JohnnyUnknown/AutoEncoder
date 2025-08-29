from tensorflow.keras.models import Model
from keras.saving import load_model
from PIL import Image
import numpy as np
from numpy.linalg import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity

from sys import path
from pathlib import Path

target_size = 256
IMAGE_DIR = Path(f"C:\\My\\Projects\\images\\main\\Data_img\\Dataset_{target_size}\\test")

def load_and_preprocess(path):
    img = Image.open(path).convert('L')
    img = img.resize((target_size, target_size))
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array


file_model = Path(path[0] + "/model_256_160p_mse_1.keras")
# file_model = Path(path[0] + "/new_model_1.keras")
autoencoder_model = load_model(file_model)
# autoencoder_model.summary()
    

# 160x160
img_path_1 = IMAGE_DIR / "g_ (4)_rotated_270.jpg"
img_path_2 = IMAGE_DIR / "g_ (7)_rotated_0_flipped_channels_permuted.jpg"
img_path_3 = IMAGE_DIR / "g_ (4)_rotated_90_channels_permuted.jpg"
img_path_4 = IMAGE_DIR / "y_ (4)_rotated_270.jpg"
img_path_5 = IMAGE_DIR / "y_ (9)_rotated_180_flipped_channels_permuted.jpg"
img_path_6 = IMAGE_DIR / "y_ (4)_rotated_0_channels_permuted.jpg"

# # 192x192
# img_path_1 = f"C:\\My\\Projects\\images\\main\\Data_img\\Dataset_{target_size}\\test\\g_ (7)_rotated_90_flipped.jpg"
# img_path_2 = f"C:\\My\\Projects\\images\\main\\Data_img\\Dataset_{target_size}\\train_0\\g_ (7)_rotated_270_channels_permuted.jpg"
# img_path_3 = f"C:\\My\\Projects\\images\\main\\Data_img\\Dataset_{target_size}\\train_1\\g_ (12)_rotated_270_channels_permuted.jpg"
# img_path_4 = f"C:\\My\\Projects\\images\\main\\Data_img\\Dataset_{target_size}\\test\\y_ (17)_rotated_270_flipped.jpg"
# img_path_5 = f"C:\\My\\Projects\\images\\main\\Data_img\\Dataset_{target_size}\\test\\y_ (12)_rotated_0.jpg"


test_img1 = load_and_preprocess(img_path_1)
test_img1 = np.expand_dims(test_img1, axis=0)
    
test_img2 = load_and_preprocess(img_path_2)
test_img2 = np.expand_dims(test_img2, axis=0)   
 
test_img3 = load_and_preprocess(img_path_3)
test_img3 = np.expand_dims(test_img3, axis=0)

test_img4 = load_and_preprocess(img_path_4)
test_img4 = np.expand_dims(test_img4, axis=0)

test_img5 = load_and_preprocess(img_path_5)
test_img5 = np.expand_dims(test_img5, axis=0)

test_img6 = load_and_preprocess(img_path_6)
test_img6 = np.expand_dims(test_img6, axis=0)



feature_extractor = Model(inputs=autoencoder_model.input, outputs=autoencoder_model.get_layer('latent_features').output)


# test_paths = [p for p in (IMAGE_DIR).glob('*')]
# test_data = [load_and_preprocess(p) for p in test_paths]
# test_data = np.array(test_data)
# print(f"\nФорма тестовых данных: {test_data.shape}")

# metrics = autoencoder_model.evaluate(test_data, test_data)
# print("\nМетрики my_model:", metrics, "\n\n")


features_1 = feature_extractor.predict(test_img1)
features_2 = feature_extractor.predict(test_img2)
features_3 = feature_extractor.predict(test_img3)
features_4 = feature_extractor.predict(test_img4)
features_5 = feature_extractor.predict(test_img5)
features_6 = feature_extractor.predict(test_img6)



distance_1 = cosine_similarity(features_1, features_3)
distance_2 = cosine_similarity(features_1, features_2)
distance_3 = cosine_similarity(features_1, features_4)
distance_4 = cosine_similarity(features_1, features_5)
distance_5 = cosine_similarity(features_4, features_5)
distance_6 = cosine_similarity(features_4, features_6)


print("Cosine:")
print("Одинаковые google изображения: ", distance_1)
print("Разные google изображения: ", distance_2)
print("Одинаковые разнородные изображения: ", distance_3)
print("Разные разнородные изображения: ", distance_4)
print("Разные yandex изображения: ", distance_5)
print("Одинаковые yandex изображения: ", distance_6, "\n")


distance_1 = mean_absolute_error(features_1, features_3)
distance_2 = mean_absolute_error(features_1, features_2)
distance_3 = mean_absolute_error(features_1, features_4)
distance_4 = mean_absolute_error(features_1, features_5)
distance_5 = mean_absolute_error(features_4, features_5)
distance_6 = mean_absolute_error(features_4, features_6)

print("MAE:")
print("Одинаковые google изображения: ", distance_1)
print("Разные google изображения: ", distance_2)
print("Одинаковые разнородные изображения: ", distance_3)
print("Разные разнородные изображения: ", distance_4)
print("Разные yandex изображения: ", distance_5)
print("Одинаковые yandex изображения: ", distance_6, "\n")


distance_1 = mean_squared_error(features_1, features_3)
distance_2 = mean_squared_error(features_1, features_2)
distance_3 = mean_squared_error(features_1, features_4)
distance_4 = mean_squared_error(features_1, features_5)
distance_5 = mean_squared_error(features_4, features_5)
distance_6 = mean_squared_error(features_4, features_6)

print("MSE:")
print("Одинаковые google изображения: ", distance_1)
print("Разные google изображения: ", distance_2)
print("Одинаковые разнородные изображения: ", distance_3)
print("Разные разнородные изображения: ", distance_4)
print("Разные yandex изображения: ", distance_5)
print("Одинаковые yandex изображения: ", distance_6, "\n")
