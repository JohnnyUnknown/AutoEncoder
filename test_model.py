from tensorflow.keras.models import Model
from keras.saving import load_model
from PIL import Image
import numpy as np

from sys import path
from pathlib import Path


def load_and_preprocess(path):
    img = Image.open(path)
    img_array = np.array(img)
    # Проверка каналов (RGB/RGBA/grayscale)
    if len(img_array.shape) == 2:  # Grayscale
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA -> RGB
        img_array = img_array[..., :3]
    return img_array.astype('float32') / 255.0


file_model = Path(path[0] + "/my_model.keras")
autoencoder_model = load_model(file_model)
    


test_img1 = load_and_preprocess("C:\\My\\Projects\\images\\main\\SasMapRes\\test\\map4_1-4.jpg")
test_img1 = np.expand_dims(test_img1, axis=0)
    
test_img2 = load_and_preprocess("C:\\My\\Projects\\images\\main\\SasMapRes\\train\\yandex1_1-4.jpg")
test_img2 = np.expand_dims(test_img2, axis=0)   
 
test_img3 = load_and_preprocess("C:\\My\\Projects\\images\\main\\SasMapRes\\005.jpg")
test_img3 = np.expand_dims(test_img3, axis=0)



feature_extractor = Model(inputs=autoencoder_model.input, outputs=autoencoder_model.get_layer('dense').output)
features_1 = feature_extractor.predict(test_img1)
features_2 = feature_extractor.predict(test_img2)
features_3 = feature_extractor.predict(test_img3)

from numpy.linalg import norm
from sklearn.metrics import mean_squared_error

distance_1 = mean_squared_error(features_1, features_2)
distance_2 = mean_squared_error(features_1, features_3)
distance_3 = mean_squared_error(features_2, features_3)

print(features_1, end="\n\n")  # (число образцов, 128)
print(features_2, end="\n\n")  # (число образцов, 128)
print(features_3, end="\n\n")  # (число образцов, 128)

print("Похожие изображения: ", distance_1)
print("1 с фото: ", distance_2)
print("2 с фото: ", distance_3)