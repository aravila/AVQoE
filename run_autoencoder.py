from models import BuildAudioModel, BuildVideoModel
import sys
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plot
import tables

def train_generator(filepath, no_train, batch_size):
    while 1:
        hdf5_file = tables.open_file(filepath, mode='r')
        for i in range(0,round(no_train/batch_size)):
            data = hdf5_file.root.train[i*batch_size:i*batch_size+batch_size]
            yield (data, data)
        hdf5_file.close()

def validation_generator(filepath, no_valid, batch_size):
    while 1:
        hdf5_file = tables.open_file(filepath, mode='r')
        for i in range(0, round(no_valid / batch_size)):
            data = hdf5_file.root.valid[i*batch_size:i*batch_size+batch_size]
            yield (data, data)
        hdf5_file.close()

def get_model(mode = 0, hdf5train = '', hdf5valid = '', no_train = 0, no_valid = 0, input_dim = 25344, batch_size = 16, nepoch = 50, early = 5):

    if (mode == 0):
        model = BuildVideoModel()
        modetype = "video"
    else:
        model = BuildAudioModel()
        modetype = "audio"

    encoder, autoencoder = model.autoencoder(input_dim)

    hist = autoencoder.fit_generator(train_generator(hdf5train, no_train, batch_size),
                            nb_epoch=nepoch,
                            samples_per_epoch=no_train,
                            validation_data=validation_generator(hdf5valid, no_valid, batch_size*(no_valid/no_train)),
                            nb_val_samples=no_valid,
                            callbacks=[])
    print(hist.history)
    encoder.save("../models/encoder_%s.hdf5"%(modetype)) 
    autoencoder.save("../models/autoencoder_%s.hdf5"%(modetype))
    fig = plot.figure()
    lossFig = "../results/auto_%s_loss.png"%(modetype)
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
        print("python run_autoencoder.py 0 \"'./vfeatures/vqoe_train.hdf5'\"  \"'./vfeatures/vqoe_valid.hdf5'\" 200000 50000 25344 25 5")
        print("Where")
        print("param1: '0' for video and '1' for audio")
        print("param2: hdf5 database path for training")
        print("param3: hdf5 database path for validation")
        print("param4: number of train instances")
        print("param5: number of validation instances")
        print("param6: input size")
        print("param7: batch size (default is 16)")
        print("param8: number of epochs (default is 50)")
        print("param9: early_stopping (default is 5)\n")
        exit()

    str = ""
    for arg in sys.argv[1:]:
        str += arg + ', '

    cmd = "get_model(" + str[:-2] + ")"
    eval(cmd)

run_model()
