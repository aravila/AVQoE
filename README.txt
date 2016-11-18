For usage:
python run_autoencoder.py

For video model:
THEAN_OFLAGS=device=gpu0 python run_autoencoder.py 0 "'/home/zahidakhtar/muse02/anderson/Video/vqoe.hdf5'" 150000 25344 16

For audio model:
THEANO_FLAGS=device=gpu0 python run_autoencoder.py 1 "'../afeatures/aqoe.hdf5'" 150000 257 32
