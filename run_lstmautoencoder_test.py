from models import BuildAudioModel, BuildVideoModel
import sys
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plot
import tables
import time

def train_generator(filepath, no_train, batch_size, time_step):
    while 1:
        hdf5_file = tables.open_file(filepath, mode='r')
        for i in range(0,round(no_train/batch_size)):
            data = hdf5_file.root.train[i*batch_size:i*batch_size+batch_size*time_step]
            data_step = data.reshape(batch_size, time_step, data.shape[1])
            yield (data_step, data)
        hdf5_file.close()

def validation_generator(filepath, no_valid, batch_size, time_step):
    while 1:
        hdf5_file = tables.open_file(filepath, mode='r')
        for i in range(0, round(no_valid / batch_size)):
            data = hdf5_file.root.valid[i*batch_size:i*batch_size+batch_size*time_step]
            data_step = data.reshape(batch_size, time_step, data.shape[1])
            yield (data_step, data)
        hdf5_file.close()

def get_input_dim(filepath):
    hdf5_file = tables.open_file(filepath, mode='r')
    return hdf5_file.root.train.shape[1]

def get_model(mode = 0, hdf5train = '', hdf5valid = '', no_train = 0, no_valid = 0, batch_size = 16, time_step = 100, nepoch = 50, early = 5):

    if (mode == 0):
        model = BuildVideoModel()
        modetype = "video"
    else:
        model = BuildAudioModel()
        modetype = "audio"

    input_dim = get_input_dim(hdf5train)

    encoder, autoencoder = model.LSTMAutoencoder(batch_size, time_step, input_dim)

    hist = autoencoder.fit_generator(train_generator(hdf5train, no_train, batch_size, time_step),
                            nb_epoch=nepoch,
                            samples_per_epoch=no_train,
                            validation_data=validation_generator(hdf5valid, no_valid, batch_size*(no_valid/no_train), time_step),
                            nb_val_samples=no_valid,
                            callbacks=[])
    print(hist.history)
    time_stamp = time.strftime("%Y-%m-%d_%H:%M")
    encoder.save("../models/encoder_%s_%s.hdf5"%(modetype, time_stamp))
    autoencoder.save("../models/autoencoder_%s_%s.hdf5"%(modetype, time_stamp))
    fig = plot.figure()
    lossFig = "../results/auto_%s_loss_%s.png"%(modetype, time_stamp)
    plot.plot(hist.history['loss'], label='Train loss')
#    plot.plot(hist.history['val_loss'], label='Valid loss')
    plot.ylabel('Loss')
    plot.xlabel('Epoches')
    plot.legend(loc='upper left')
    #plot.show()
    fig.savefig(lossFig)


def run_model():
    if(len(sys.argv) < 7):
        print("\nUsage:")
        print("python run_autoencoder.py 0 \"'./vfeatures/vqoe_train.hdf5'\"  \"'./vfeatures/vqoe_valid.hdf5'\" 200000 50000 16 100 50 5")
        print("Where")
        print("param1: '0' for video and '1' for audio")
        print("param2: hdf5 database path for training")
        print("param3: hdf5 database path for validation")
        print("param4: number of train instances")
        print("param5: number of validation instances")
        print("param6: batch size (default is 16)")
        print("param7: time steps (default is 100)")
        print("param8: number of epochs (default is 50)")
        print("param9: early_stopping (default is 5)\n")
        exit()

    str = ""
    for arg in sys.argv[1:]:
        str += arg + ', '

    cmd = "get_model(" + str[:-2] + ")"
    eval(cmd)

run_model()
