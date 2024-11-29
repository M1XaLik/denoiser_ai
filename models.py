from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, BatchNormalization, ReLU

def build_autoencoder(input_shape=(700, 500, 3)):
    inputs = Input(shape=input_shape)
    # Енкодер
    x = Conv2D(64, (3, 3), strides=2, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    # Декодер
    x = Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    return Model(inputs, outputs)
