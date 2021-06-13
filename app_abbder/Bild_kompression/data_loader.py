import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, losses, regularizers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.preprocessing import image
import zipfile
import matplotlib.pyplot as plt
import cv2


print(os.getcwd())

'''
# we took the directory of the .zip file
local_zip = 'Data\cats_and_dogs_filtered\cats_and_dogs_filtered/cats_and_dogs_filtered.zip'
# We create the instance of ZipFile providing the directory.
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/Data/dataset')
zip_ref.close()
'''

img_height, img_width = (256,256)



batch_size = 8
#data_dir = "Data\kagglecatsanddogs_3367a\PetImages"

train_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last')
train_generator = train_datagen.flow_from_directory('C:/cats_and_dogs_filtered/train',
                                                    target_size=(img_height, img_width),
                                                    shuffle=True,
                                                    batch_size=batch_size,
                                                    class_mode='input')

test_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_last')
validation_generator = test_datagen.flow_from_directory(
    'C:/cats_and_dogs_filtered/validation/',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='input'
    )

#this function take as parameter an Imagedatagenerator and convert all image in
# generator into numpy array
def convert_to_nparray(data_generator):
    data_list = []
    batch_index = 0

    while batch_index <= data_generator.batch_index:
        data = data_generator.next()
        data_list.append( data[0] )
        batch_index = batch_index + 1

    # now, data_array is the numeric data of whole images
    data_array = np.asarray( data_list )
    return data_array

def show_image(x, n=10, title=''):
    plt.figure(figsize=(15,5))
    for i in range(n):
        ax=plt.subplot(2, n, i+1)
        plt.imshow(image.array_to_img(x[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)




#test model
autoencoder_middel = tf.keras.Sequential()
autoencoder_middel.add(keras.Input( shape=(256, 256, 3) )),

autoencoder_middel.add(layers.Conv2D(64, (3, 3), padding='same',activation='relu', input_shape=(224,224,3)))
autoencoder_middel.add(layers.MaxPooling2D( pool_size=(2, 2) ))

autoencoder_middel.add(layers.Conv2D(32, (3, 3), padding='same',activation='relu', input_shape=(224,224,3)))
autoencoder_middel.add(layers.MaxPooling2D( pool_size=(2, 2) ))

autoencoder_middel.add(layers.Conv2D(16, (3, 3), padding='same',activation='relu', input_shape=(224,224,3)))
autoencoder_middel.add(layers.MaxPooling2D( pool_size=(2, 2) ))

autoencoder_middel.add(layers.Conv2D(3, (3, 3), padding='same',activation='relu', input_shape=(224,224,3)))

autoencoder_middel.add(layers.Conv2DTranspose(16, (3, 3), padding='same',activation='relu', input_shape=(224,224,3)))
autoencoder_middel.add(layers.UpSampling2D( (2, 2) ))

autoencoder_middel.add(layers.Conv2DTranspose(32, (3, 3), padding='same',activation='relu', input_shape=(224,224,3)))
autoencoder_middel.add(layers.UpSampling2D( (2, 2) ))

autoencoder_middel.add(layers.Conv2DTranspose(64, (3, 3), padding='same',activation='relu', input_shape=(224,224,3)))
autoencoder_middel.add(layers.UpSampling2D( (2, 2) ))


autoencoder_middel.add(layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid'))

autoencoder_middel.summary()


#config the compiler

autoencoder_middel.compile(optimizer='adamax', loss='mse', metrics=["accuracy"])

#train the model
history = autoencoder_middel.fit_generator(
    train_generator,
    steps_per_epoch= 1000//8,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=1000/8
)

data = train_generator.next()

predicted = autoencoder_middel.predict(data[0])
plt.imshow(predicted[0])



#       Methode 1
#   using dataset from directory

'''
img_height = 0
img_width = 0
img_size = 0
batchsize = 2

ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    ' Data/ ',
)
'''