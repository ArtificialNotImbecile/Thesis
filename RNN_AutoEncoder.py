from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed,Masking
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.neighbors import kneighbors_graph

class RNN_AutoEncoder(object):
    """
        RNN AE with only embedding layer contains lstm cell
    """
    def __init__(self, activation='sigmoid'):
        self.activation = activation

    def compile(self, optimizer='ada', loss='categorial_crossentropy'):
        model = Sequential()
        #model.add(Masking(mask_value=0.,input_shape=(None,5)))
        model.add(TimeDistributed(Dense(32, activation=self.activation),input_shape=(None,784)))
        model.add(LSTM(16, return_sequences=True))
        model.add(TimeDistributed(Dense(32, activation=self.activation)))
        model.add(TimeDistributed(Dense(784, activation=self.activation)))
        print(model.summary())
        self.model = model
        self.model.compile(loss='categorical_crossentropy',
              optimizer='adam')

    def fit_generator(self, train_generator, steps_per_epoch=30, epochs=100, verbose=1):
        self.model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=verbose)

    def comparison_plot(self, data, _i=0):
        fig, axe = plt.subplots(2,4, figsize=(7,7))
        ipt = data[_i].reshape(4,28,28)
        new_x = np.array(data)[_i=0].reshape(1,4,784)
        pred = self.model.predict(new_x).reshape((4,28,28))
        plt.gray()
        for i in range(4):
            axe[0,i].imshow(1-pred[i])
            axe[1,i].imshow(1-ipt[i])
        plt.show()

def get_rnn_data(num_data=100,index=0):
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    data = mnist.test.images # Returns np.array
    labels = np.asarray(mnist.test.labels, dtype=np.int32)

    zero_index = labels == index
    data_0 = data[zero_index]
    labels_0 = labels[zero_index]
    # produce KNN pairs (data should have three dimension:[num_samples,timestep,dim]==[100,4,784]) we can only use one class a time? yeah...
    np.random.shuffle(data_0)
    data_knn = data_0[:num_data]
    G = kneighbors_graph(data_knn, 4, mode='connectivity',include_self=True,n_jobs=-1).toarray()
    data_pairs_knn3 = [data_knn[G[i,:]==1] for i in range(num_data)]
    print(np.shape(data_pairs_knn3))
    return data_pairs_knn3
