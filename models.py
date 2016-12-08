from keras.layers import Input, Dense, LSTM, RepeatVector
from keras.models import Model
from keras.optimizers import RMSprop

class BuildAudioModel(object):

    def __init__(self):
        print("Build DNN Model")

    def autoencoder(self, input_dim, hid_factor = 2):

        hid_dim = round(input_dim/hid_factor)
        input_val = Input(shape=(input_dim,))
        encoded = Dense(hid_dim, activation='relu')(input_val)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)

        autoencoder = Model(input=input_val, output=decoded)

        encoder = Model(input=input_val, output=encoded)

        rms = RMSprop(lr=0.001)

        autoencoder.compile(optimizer=rms, loss='mean_squared_error')

        return encoder, autoencoder


    def LSTMAutoencoder_test(self, batch_size, timesteps, input_dim, hid_factor = 2):

        input_val = Input(batch_input_shape=(None, timesteps, input_dim))
        encoded = LSTM(128)(input_val)

        decoded = RepeatVector(timesteps)(encoded)
        decoded = LSTM(input_dim, return_sequences=True)(decoded)

        autoencoder = Model(input=input_val, output=decoded)
        encoder = Model(input=input_val, output=encoded)

        rms = RMSprop(lr=0.001)

        autoencoder.compile(optimizer=rms, loss='mean_squared_error')

        return encoder, autoencoder

    def LSTMAutoencoder(self, timesteps, input_dim, hid_factor = 2):

        input_val = Input(shape=(timesteps, input_dim))
        encoded = LSTM(128)(input_val)

        decoded = RepeatVector(timesteps)(encoded)
        decoded = LSTM(input_dim, return_sequences=True)(decoded)

        autoencoder = Model(input=input_val, output=decoded)
        encoder = Model(input=input_val, output=encoded)

        rms = RMSprop(lr=0.001)

        autoencoder.compile(optimizer=rms, loss='mean_squared_error')

        return encoder, autoencoder


    def deepautoencoder(self, input_dim, hid_factor=2):

        hid_dim = round(input_dim / hid_factor)
        input_val = Input(shape=(input_dim,))
        encoded = Dense(hid_dim, activation='relu')(input_val)
        encoded = Dense(hid_dim, activation='relu')(encoded)

        decoded = Dense(hid_dim, activation='relu')(encoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)

        autoencoder = Model(input=input_val, output=decoded)

        encoder = Model(input=input_val, output=encoded)

        rms = RMSprop(lr=0.01)

        autoencoder.compile(optimizer=rms, loss='mean_squared_error')

        return encoder, autoencoder


class BuildVideoModel(object):
    def __init__(self):
        print("Build DNN Model")

    def autoencoder(self, input_dim, hid_factor = 10):

        hid_dim = round(input_dim/hid_factor)
        input_val = Input(shape=(input_dim,))
        encoded = Dense(hid_dim, activation='relu')(input_val)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)

        autoencoder = Model(input=input_val, output=decoded)

        encoder = Model(input=input_val, output=encoded)

        rms = RMSprop(lr=0.001)

        autoencoder.compile(optimizer=rms, loss='mean_squared_error')

        return encoder, autoencoder


    def Convautoencoder(self, input_dim, hid_factor = 10):

        hid_dim = round(input_dim/hid_factor)
        input_val = Input(shape=(input_dim,))
        encoded = Dense(hid_dim, activation='relu')(input_val)
        decoded = Dense(input_dim, activation='sigmoid')(encoded)

        autoencoder = Model(input=input_val, output=decoded)

        encoder = Model(input=input_val, output=encoded)

        rms = RMSprop(lr=0.001)

        autoencoder.compile(optimizer=rms, loss='mean_squared_error')

        return encoder, autoencoder


    def deepautoencoder(self, input_dim, hid_factor=10):

        hid_dim = round(input_dim / hid_factor)
        input_val = Input(shape=(input_dim,))
        encoded = Dense(hid_dim, activation='relu')(input_val)

        decoded = Dense(hid_dim, activation='relu')(encoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)

        autoencoder = Model(input=input_val, output=decoded)

        encoder = Model(input=input_val, output=encoded)

        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        return encoder, autoencoder
