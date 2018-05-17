import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
import matplotlib.pyplot as plt

class AutoEncoder(object):
    def __init__(self, layers=[784, 32, 784], activation='relu', final_actication='sigmoid',regularize=False):
        self.num_layers = len(layers)
        self.layers = layers
        assert self.num_layers%2!=0
        self.activation = activation
        self.regularize = regularize
        self.final_actication = final_actication

    def compile(self, optimizer='adadelta', loss='binary_crossentropy'):
        if self.regularize:
            #encoder part
            input_img = Input(shape=(self.layers[0],))
            encoded = Dense(self.layers[1], activation=self.activation, activity_regularizer=regularizers.l1(10e-5))(input_img)
            for i in range(int((self.num_layers-3)/2)):
                encoded = Dense(self.layers[i+2], activation=self.activation, activity_regularizer=regularizers.l1(10e-5))(encoded)
            # decoder part
            if self.num_layers != 3:
                decoded = Dense(self.layers[int((self.num_layers+1)/2)], activation=self.activation, activity_regularizer=regularizers.l1(10e-5))(encoded)
            for i in range(int((self.num_layers-3)/2-1)):
                j = i + int((self.num_layers+3)/2)
                decoded = Dense(self.layers[j], activation=self.activation, activity_regularizer=regularizers.l1(10e-5))(decoded)
            # ???????????????is this correct????????????????sigmoid? and don't use regularizers?
            if self.num_layers !=3:
                decoded = Dense(self.layers[-1], activation=self.final_actication)(decoded)
            else:
                decoded = Dense(self.layers[-1], activation=self.final_actication)(encoded)
            self.autoencoder = Model(input_img, decoded)
            self.encoder = Model(input_img, encoded)
            # Decoder is a little bit tricker
            encoded_input = Input(shape=(self.layers[int((self.num_layers-1)/2)],))
            # retrieve the last layer of the autoencoder model
            decoder_layer = self.autoencoder.layers[int((self.num_layers+1)/2)](encoded_input)
            for i in range(int((self.num_layers-3)/2)):
                j = i + int((self.num_layers+3)/2)
                decoder_layer = self.autoencoder.layers[j](decoder_layer)
            # create the decoder model
            self.decoder = Model(encoded_input, decoder_layer)
            self.autoencoder.compile(optimizer=optimizer, loss=loss)
        else:
            #encoder part
            input_img = Input(shape=(self.layers[0],))
            encoded = Dense(self.layers[1], activation=self.activation)(input_img)
            for i in range(int((self.num_layers-3)/2)):
                encoded = Dense(self.layers[i+2], activation=self.activation)(encoded)
            # decoder part
            if self.num_layers!=3:
                decoded = Dense(self.layers[int((self.num_layers+1)/2)], activation=self.activation)(encoded)
            for i in range(int((self.num_layers-3)/2-1)):
                j = i + int((self.num_layers+3)/2)
                decoded = Dense(self.layers[j], activation=self.activation)(decoded)
            # ???????????????is this correct????????????????
            if self.num_layers !=3:
                decoded = Dense(self.layers[-1], activation=self.final_actication)(decoded)
            else:
                decoded = Dense(self.layers[-1], activation=self.final_actication)(encoded)
            self.autoencoder = Model(input_img, decoded)
            self.encoder = Model(input_img, encoded)
            # Decoder is a little bit tricker
            encoded_input = Input(shape=(self.layers[int((self.num_layers-1)/2)],))
            # retrieve the last layer of the autoencoder model
            decoder_layer = self.autoencoder.layers[int((self.num_layers+1)/2)](encoded_input)
            for i in range(int((self.num_layers-3)/2)):
                j = i + int((self.num_layers+3)/2)
                decoder_layer = self.autoencoder.layers[j](decoder_layer)
            # create the decoder model
            self.decoder = Model(encoded_input, decoder_layer)
            self.autoencoder.compile(optimizer=optimizer, loss=loss)

    def fit(self, x, y, epochs=100, batch_size=256, shuffle=True, **kwarg):
        self.autoencoder.fit(x, y,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=shuffle,
                **kwarg)

    def get_encoded_imgs(self, x_new):
        return self.encoder.predict(x_new)

    def get_decoded_imgs(self, x_new):
        encoded_imgs = self.encoder.predict(x_new)
        return self.decoder.predict(encoded_imgs)

    def comparison_plot(self, x_new, n=10, figsize=(20,4)):
        plt.figure(figsize=(20, 4))
        decoded_imgs = self.get_decoded_imgs(x_new)
        for i in range(n):
            # display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(1-x_new[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(1-decoded_imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

def add_noise_to_data(x, noise_factor=0.5,clip=False):
        x_noisy = x + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x.shape)
        if clip:
            return np.clip(x_noisy, 0., 1.)
        else:
            return x_noisy
