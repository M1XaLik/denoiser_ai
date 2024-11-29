import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_loader import ImageDataGenerator
from models import build_autoencoder

def train_model():
    # Шляхи до даних
    damaged_path = 'dataset/training_dts/damaged/'
    original_path = 'dataset/training_dts/original/'
    valid_damaged_path = 'dataset/valid_dts/damaged/'
    valid_original_path = 'dataset/valid_dts/original/'

    # Параметри
    batch_size = 16
    img_size = (500, 700)

    # Генератори
    train_data = ImageDataGenerator(damaged_path, original_path, batch_size, img_size)
    val_data = ImageDataGenerator(valid_damaged_path, valid_original_path, batch_size, img_size)

    # Модель
    model = build_autoencoder()
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    # Колбеки
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint('saved_model/autoencoder.keras', save_best_only=True)
    ]

    # Навчання
    model.fit(
        train_data.generate(),
        validation_data=val_data.generate(),
        steps_per_epoch=len(train_data),
        validation_steps=len(val_data),
        epochs=10,
        callbacks=callbacks
    )

if __name__ == '__main__':
    train_model()
