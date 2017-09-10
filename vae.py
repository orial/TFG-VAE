''' Autoencoder Variacional con TensorFlow y Keras. Adaptado para TFG Lupicinio García Ortiz
Referencia: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''

import numpy as np
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import backend as K
from keras import metrics

import time

# dataset de entrada
entrada = "GasSensors.csv"
salida = "GasSensors_red.csv"
original_dim = 128

# Load CSV data in a format suited for tensorflow
dataset = np.loadtxt(entrada, delimiter=",")

# split into input (set_x) and output (set_y) variables
set_x = dataset[:,0:original_dim]
set_y = dataset[:,original_dim]

batch_size = 100
latent_dim = 2
intermediate_dim = 50
intermediate_dim1 = 50
epochs = 20
epsilon_std = 0.01

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
h1 = Dense(intermediate_dim1, activation='relu')(h)

#z_mean = Dense(latent_dim)(h)
z_mean = Dense(latent_dim)(h1)

#z_log_var = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h1)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_h1 = Dense(intermediate_dim1, activation='relu')

decoder_mean = Dense(original_dim, activation='sigmoid')

h_decoded = decoder_h(z)
h1_decoded = decoder_h1(z)

#x_decoded_mean = decoder_mean(h_decoded)
x_decoded_mean = decoder_mean(h1_decoded)

# Custom loss layer
class CustomVariationalLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_decoded_mean):
        xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        x_decoded_mean = inputs[1]
        loss = self.vae_loss(x, x_decoded_mean)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x

y = CustomVariationalLayer()([x, x_decoded_mean])
vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=None)

x_train = set_x
x_test  = set_x

start = time.time()

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

end = time.time()
runtime = '{:.3f}'.format(end-start) 

print('Total runtime over the training set:', runtime)

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)


x_test_encoded = encoder.predict(set_x, batch_size=batch_size)
test_encoded = np.insert(x_test_encoded, latent_dim, set_y, axis=1)
np.savetxt(salida, test_encoded, delimiter=",")
