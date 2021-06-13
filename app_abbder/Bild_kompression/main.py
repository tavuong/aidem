import matplotlib as plt
from tensorflow.keras.datasets import fashion_mnist, cifar10
from tensorflow.keras import layers, losses, regularizers
from ae_model import Autoencoder
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def load_cifar():
    (x_train, _), (x_test, _) = cifar10.load_data()
    x_train = x_train.astype( 'float32' ) / 255.
    x_test = x_test.astype( 'float32' ) / 255.
    return x_train, x_test

def load_fashion():
    (x_train, _), (x_test, _) = fashion_mnist.load_data()
    x_train = x_train.astype( 'float32' ) / 255.
    x_test = x_test.astype( 'float32' ) / 255.
    return x_train, x_test


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # first Load Data
    (x_train, _), (x_test, _) = cifar10.load_data()
    # Normalize the train Data
    x_train = x_train.astype( 'float32' ) / 255.
    # Normalize the test Data
    x_test = x_test.astype( 'float32' ) / 255.

    print( " data train shape" )
    print( x_train.shape )# shape ist 32*32*3
    print( " data test shape" )
    print( x_test.shape )# shape ist 32*32*3

    #initialize the Autoencoder

    autoencoder = Autoencoder()
    #Load Weight
    autoencoder.load_weights( "saved_model/" )
    #config the model
    autoencoder.compile( optimizer='adam', loss=losses.MeanSquaredError(), metrics=["accuracy"] )
    # train the model
    # history= autoencoder.fit(x_train, x_train, epochs=1, batch_size = 10,  shuffle=True, validation_data=(x_test, x_test))

    # we take 10 Validation data to test the model
    a_test = x_test[80:91]
    # we encode
    encoded_imgs = autoencoder.encoder( a_test ).numpy()
    print(encoded_imgs)
    decoded_imgs = autoencoder.decoder( encoded_imgs ).numpy()

    # encoded_imgs1 = autoencoder.encoder(a_test).size()

    autoencoder.evaluate( x_test, x_test, verbose=2, batch_size=32, )

    autoencoder.save_weights( "saved_model/" )
    print( "autoencoder summary" )
    autoencoder.summary()
    print( "encoder summayry" )
    autoencoder.encoder.summary()

    print( "decoder summayry" )
    autoencoder.decoder.summary()

    """# **Testing on model**"""

    # Display and compare the Original Image with the reconstructed image

    n = 10
    plt.figure( figsize=(20, 4) )
    for i in range( n ):
        # display original
        ax = plt.subplot( 2, n, i + 1 )
        plt.imshow( a_test[i] )
        plt.title( "original" )

        ax.get_xaxis().set_visible( False )
        ax.get_yaxis().set_visible( False )

        # display reconstruction
        ax = plt.subplot( 2, n, i + 1 + n )
        plt.imshow( decoded_imgs[i] )
        plt.title( "reconst" )

        ax.get_xaxis().set_visible( False )
        ax.get_yaxis().set_visible( False )
    plt.show()


