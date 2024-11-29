import cv2
import numpy as np
from tensorflow.keras.models import load_model

def denoise_image(input_path, output_path, model_path='saved_model/autoencoder.keras'):
    # Завантаження моделі
    model = load_model(model_path)

    # Завантаження зображення
    original_img = cv2.imread(input_path)
    original_shape = original_img.shape[:2]  # Зберегти початкові розміри (height, width)

    # Зміна розміру до розміру, який очікує модель
    resized_img = cv2.resize(original_img, (500, 700))
    input_data = resized_img.astype('float32') / 255.0  # Нормалізація
    input_data = np.expand_dims(input_data, axis=0)  # Додавання batch dimension

    # Прогноз моделі
    output_data = model.predict(input_data)[0]

    # Переведення з нормалізованого формату [0, 1] назад до [0, 255]
    output_data = (output_data * 255.0).astype('uint8')

    # Масштабування назад до початкового розміру
    output_img = cv2.resize(output_data, (original_shape[1], original_shape[0]))

    # Збереження зображення
    cv2.imwrite(output_path, output_img)

    print(f"Відновлене зображення збережено як {output_path}")

if __name__ == '__main__':
    # Використання
    denoise_image('input.jpg', 'output.jpg')
