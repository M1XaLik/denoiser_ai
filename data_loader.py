import os
import cv2
import numpy as np

class ImageDataGenerator:
    def __init__(self, damaged_path, original_path, batch_size=32, img_size=(700, 500)):
        self.damaged_path = damaged_path
        self.original_path = original_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.damaged_files = sorted(os.listdir(damaged_path))
        self.original_files = sorted(os.listdir(original_path))

    def __len__(self):
        # Повертає кількість батчів
        return len(self.damaged_files) // self.batch_size

    def __getitem__(self, idx):
        # Отримати індекси для батчу
        batch_damaged_files = self.damaged_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_original_files = self.original_files[idx * self.batch_size:(idx + 1) * self.batch_size]

        x = []
        y = []

        for damaged_file, original_file in zip(batch_damaged_files, batch_original_files):
            damaged_img = cv2.resize(cv2.imread(os.path.join(self.damaged_path, damaged_file)), self.img_size) / 255.0
            original_img = cv2.resize(cv2.imread(os.path.join(self.original_path, original_file)), self.img_size) / 255.0
            x.append(damaged_img)
            y.append(original_img)

        return np.array(x), np.array(y)

    def generate(self):
        # Генератор для ітеративного повернення батчів
        while True:
            for i in range(self.__len__()):
                yield self[i]
