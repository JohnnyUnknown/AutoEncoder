from tensorflow.keras.models import Model
from keras.saving import load_model
from PIL import Image
import numpy as np
from numpy.linalg import norm
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sys import path
from pathlib import Path

IMAGE_DIR = Path("C:\\My\\Projects\\images\\main\\Data_img\\Dataset\\test")

def load_and_preprocess(path):
    img = Image.open(path).convert('L')
    img = img.resize((320, 320))
    img_array = np.array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array


file_model = Path(path[0] + "/my_model_0.keras")
# file_model = Path(path[0] + "/my_base_model.keras")
autoencoder_model = load_model(file_model)
autoencoder_model.summary()
    


test_img1 = load_and_preprocess(IMAGE_DIR / "g_ (1)_rotated_0.jpg")
test_img1 = np.expand_dims(test_img1, axis=0)
    
test_img2 = load_and_preprocess(IMAGE_DIR / "y_ (1)_rotated_0_flipped.jpg")
test_img2 = np.expand_dims(test_img2, axis=0)   
 
test_img3 = load_and_preprocess(IMAGE_DIR / "g_ (7)_rotated_0_flipped.jpg")
test_img3 = np.expand_dims(test_img3, axis=0)



feature_extractor = Model(inputs=autoencoder_model.input, outputs=autoencoder_model.get_layer('latent_features').output)


test_paths = [p for p in (IMAGE_DIR).glob('*')]
test_data = [load_and_preprocess(p) for p in test_paths]
test_data = np.array(test_data)
print(f"Форма тестовых данных: {test_data.shape}")

metrics = autoencoder_model.evaluate(test_data, test_data)
print(metrics, "\n\n")


features_1 = feature_extractor.predict(test_img1)
features_2 = feature_extractor.predict(test_img2)
features_3 = feature_extractor.predict(test_img3)


distance_1 = mean_squared_error(features_1, features_2)
distance_2 = mean_squared_error(features_1, features_3)
distance_3 = mean_squared_error(features_2, features_3)


print("Похожие изображения (MSE): ", distance_1)
print("Непохожее 1 изображения: ", distance_2)
print("Непохожие 2 изображения: ", distance_3, "\n")

distance_1 = mean_absolute_error(features_1, features_2)
distance_2 = mean_absolute_error(features_1, features_3)
distance_3 = mean_absolute_error(features_2, features_3)

print("Похожие изображения (MAE): ", distance_1)
print("Непохожее 1 изображения: ", distance_2)
print("Непохожие 2 изображения: ", distance_3)

