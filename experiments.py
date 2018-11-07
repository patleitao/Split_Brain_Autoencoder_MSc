from keras.layers import *
from keras.models import Model
from keras.constraints import maxnorm
from keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard
import numpy as np
import scipy
from autoencoder import *
from utils import *
import sys


def run_bw_autoencoder(fname, epochs=20):
	imgs = load(fname)
	imgs = imgs[:, :, :, np.newaxis]
	train_imgs = imgs[:1600]
	test_imgs  = imgs[1600:]
	train_imgs = normalise(train_imgs)
	test_imgs  = normalise(train_imgs)

	model = Autoencoder(train_imgs, train_imgs, test_imgs, test_imgs)

	model.train(epochs=epochs)
	save(model, 'trad_ac_model')
	model.plot_results()

def test_save_load_model(fname):
	imgs = load(fnaem)
	model = Autoencoder()



def main():
	fname = ''
	save_name = 'test_results'
	epochs = 10
	if len(sys.argv) >=2:
		train_name = sys.argv[1]
	if len(sys.argv) >=3:
		test_name = sys.argv[2]
	if len(sys.argv)>=4:
		save_name = sys.argv[3]
	if len(sys.argv)>=5:
		epochs = int(sys.argv[4])
	if len(sys.argv) <=1:
		raise ValueError('Need to input a filename for the data when running the model')

	run_bw_autoencoder(train_name)

if __name__ == '__main__':
	main()