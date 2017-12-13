import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator

def get_generator(path, batch_size):
    img_size = 224
    train_gen = ImageDataGenerator(samplewise_center=True,
                                   samplewise_std_normalization=True,
					rotation_range=20,
					width_shift_range=50,
					height_shift_range=50)
    
    val_gen = ImageDataGenerator(samplewise_center=True,
                                   samplewise_std_normalization=True)
    
    train  = train_gen.flow_from_directory(path + '/train',
                                           target_size=(img_size, img_size),
                                          batch_size=batch_size)
    val = val_gen.flow_from_directory(path + '/val',
                                      target_size=(img_size, img_size),
                                     batch_size=batch_size)
    return train, val

