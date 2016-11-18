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

def validation_generator(filepath, no_train, no_valid, batch_size):
    while 1:
        hdf5_file = tables.open_file(filepath, mode='r')
        #for i in range(round(no_train/batch_size) + 1,round((no_train)/batch_size) + round((no_valid)/batch_size) + 1):
        for i in range(0, round(no_valid / batch_size)):
            data = hdf5_file.root.valid[i*batch_size:i*batch_size+batch_size]
            yield (data, data)
        hdf5_file.close()

def get_model(mode = 0, hdf5file = '', no_train = 0, no_valid = 0, input_dim = 25344, batch_size = 16, nepoch = 50, early = 5):

    if (mode == 0):
        model = BuildVideoModel()
        modetype = "video"
    else:
        model = BuildAudioModel()
        modetype = "audio"

    encoder, autoencoder = model.autoencoder(input_dim)

    hdf5train = '/home/zahidakhtar/muse02/anderson/Video/vqoe_test.hdf5'
    hdf5valid = '/home/zahidakhtar/muse02/anderson/Video/vqoe_test.hdf5'
    hist = autoencoder.fit_generator(train_generator(hdf5train, no_train, batch_size),
                            nb_epoch=nepoch,
                            samples_per_epoch=no_train,
                            validation_data=validation_generator(hdf5valid, no_train, no_valid, batch_size/2),
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
    if(len(sys.argv) < 6):
        print("\nUsage:")
        print("python run_autoencoder.py 0 \"'./vfeatures/vqoe.hdf5'\" 200000 40000 25344")
        print("Where")
        print("param1: '0' for video and '1' for audio")
        print("param2: hdf5 database path")
        print("param3: number of train instances")
        print("param4: number of validation instances")
        print("param5: input size")
        print("param6: batch size (default is 16)")
        print("param7: number of epochs (default is 50)")
        print("param8: early_stopping (default is 5)\n")
        exit()

    str = ""
    for arg in sys.argv[1:]:
        str += arg + ', '

    cmd = "get_model(" + str[:-2] + ")"
    eval(cmd)

run_model()
