import os
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

# Пути
input_dir_g = Path(r"C:\My\Projects\images\main\Data_img\google_aug")
input_dir_y = Path(r"C:\My\Projects\images\main\Data_img\yandex_aug")
input_dir_img = Path(r"C:\My\Projects\images\main\Data_img\images_aug")
output_dir = Path(r"C:\My\Projects\images\main\Data_img\Dataset_imgs")

# Размер для изменения
target_size = (320, 320)  # (ширина, высота)

# Поддерживаемые расширения
supported_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# Создаём папки для train и test, если их нет
for i in range(2):
    (train_dir := output_dir / f'train{i}').mkdir(parents=True, exist_ok=True)
(test_dir := output_dir / 'test').mkdir(parents=True, exist_ok=True)

# Получаем список путей к изображениям с поддерживаемыми расширениями
# image_paths_g = [p for p in input_dir_g.iterdir() if p.suffix.lower() in supported_ext]
# image_paths_y = [p for p in input_dir_y.iterdir() if p.suffix.lower() in supported_ext]
# image_paths = image_paths_g + image_paths_y

image_paths = [p for p in input_dir_img.iterdir() if p.suffix.lower() in supported_ext]

if not image_paths:
    raise FileNotFoundError(f"В папке не найдено изображений с расширениями {supported_ext}")

# Разбиваем на train и test
train_paths, test_paths = train_test_split(image_paths, test_size=0.1, random_state=42)

def process_and_save(paths, save_dir):
    for img_path in paths:
        try:
            with Image.open(img_path) as img:
                img = img.resize(target_size)
                save_path = save_dir / img_path.name
                img.save(save_path)
                # print(f"Сохранено: {save_path}")
        except Exception as e:
            print(f"Ошибка обработки {img_path}: {e}")

def process_and_save_train(paths):
    cnt = 0
    for img_path in paths:
        cnt += 1
        try:
            with Image.open(img_path) as img:
                img = img.resize(target_size)
                save_path = (output_dir / f'train{cnt // 450}') / img_path.name
                img.save(save_path)
                # print(f"Сохранено: {save_path}")
        except Exception as e:
            print(f"Ошибка обработки {img_path}: {e}")

# Обрабатываем и сохраняем
process_and_save_train(train_paths)
process_and_save(test_paths, test_dir)

print("Готово.")
