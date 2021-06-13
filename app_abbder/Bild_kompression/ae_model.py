import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, regularizers, Model
from tensorflow.keras import layers




###### Sub Class Model

class Autoencoder(Model):
  def __init__(self, ):
    super(Autoencoder, self).__init__()
    self.encoder = tf.keras.Sequential([
        keras.Input( shape=(32, 32, 3) ),

        layers.Conv2D( 64, 3, activation='relu', padding='same' ),
        layers.MaxPooling2D( pool_size=(2, 2) ),#16*16*64

        layers.Conv2D( 32, 3, activation='relu', padding='same' ),
        layers.MaxPooling2D( pool_size=(2, 2) ),#8*8*32

    ])
    self.decoder = tf.keras.Sequential([
        layers.Conv2D( 32, 3, activation='relu', padding='same' ),
        layers.UpSampling2D( (2, 2) ),

        layers.Conv2D( 64, 3, activation='relu', padding='same' ),
        layers.UpSampling2D( (2, 2) ),

        layers.Conv2D( 3, 3, activation='sigmoid', padding='same' ),
    ])

  def load_param(self, model_name):
    self.load_weights(model_name+'/')

  def save_model(self, model_name):
    self.save_weights( model_name+"/" )

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


###########   Sequential model
model = keras.Sequential( [

    keras.Input(shape=(32, 32 ,3)),

    layers.Conv2D(64, 3, activation='relu', padding='same' ),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Conv2D(32, 3, activation='relu', padding='same' ),
    layers.MaxPooling2D(pool_size=(2,2)),

    layers.Conv2D(8, 3, activation='relu', padding='same' ),
    layers.MaxPooling2D(pool_size=(2,2)),


    layers.Conv2D(8, 3, activation='relu', padding='same' ),
    layers.UpSampling2D((2,2)),

    layers.Conv2D(32, 3, activation='relu', padding='same' ),
    layers.UpSampling2D((2,2)),

    layers.Conv2D(64, 3, activation='relu', padding='same' ),
    layers.UpSampling2D((2,2)),

    layers.Conv2D(3, 3, activation='sigmoid', padding='same' ),

    ])

#print(model.summary())

# model konfigurieren
'''model.compile(optimizer=keras.optimizer.Adam(lr=0.001),
                loss=losses.MeanSquaredError(),
                metrics=["accuracy"])
model.compile(optimizer='adam', loss=losses.MeanSquaredError())
model.fit(x_train, x_train, epochs=10, shuffle=True, validation_data=(x_test, x_test))
'''

