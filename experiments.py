from autoencoder import *
from utils import *
import sys
import math


def run_autoencoder(fname, sname, epochs=15):
    imgs = load_array(fname)
    imgs = imgs[:, :, :, np.newaxis]
    train_size = int(math.ceil(len(imgs) * 85 / 100))
    train_imgs = imgs[:train_size]
    test_imgs  = imgs[train_size:]
    train_imgs = normalise(train_imgs)
    test_imgs  = normalise(test_imgs)

    ae_model = Autoencoder(train_imgs, train_imgs, test_imgs, test_imgs)

    ae_model.train(epochs=epochs)
    model = ae_model.model
    save_model(model, 'trad_ac_model')
    ae_model.plot_results()

    error_maps = ae_model.get_error_maps()

    sal_maps = load_array(sname)[train_size:]
    print('saliences loaded')

    compare_saliences(error_maps, sal_maps, show=True)

def run_sketch_autoencoder():
    imgs = load_array('Data-Sketch_images')
    imgs = imgs[:, :, :, np.newaxis]
    train_size = int(math.ceil(len(imgs) * 90 / 100))
    train_imgs = imgs[:train_size]
    test_imgs = imgs[train_size:]
    train_imgs = normalise(train_imgs)
    test_imgs = normalise(test_imgs)

    model = load_model('trad_sketch_model')
    ae_model = Autoencoder(train_imgs, train_imgs, test_imgs, test_imgs, model=model)

    ae_model.train(epochs=4)
    model = ae_model.model
    save_model(model, 'trad_sketch_model')
    ae_model.plot_results(N=7)

    error_maps = ae_model.get_error_maps()

    sal_maps = load_array('Data-Sketch_fixations')[train_size:]

    compare_saliences(error_maps, sal_maps, show=True)

def run_trained_autoencoder(fname, mname, sname, train=True):
    imgs = load_array(fname)
    imgs = imgs[:, :, :, np.newaxis]
    train_size = int(math.ceil(len(imgs) * 85 / 100))
    train_imgs = imgs[:train_size]
    test_imgs = imgs[train_size:]
    train_imgs = normalise(train_imgs)
    test_imgs = normalise(test_imgs)

    model = load_model(mname)
    ae_model = Autoencoder(train_imgs, train_imgs, test_imgs, test_imgs, model = model)

    if train:
        ae_model.train(epochs=20)
        model = ae_model.model
        save_model(model, 'trad_ac_model')

    ae_model.plot_results()
    error_maps = ae_model.get_error_maps()

    sal_maps = load_array(sname)[train_size:]
    print('saliences loaded')

    error_maps, sal_maps = shuffle_in_unison(error_maps, sal_maps)
    compare_two_images(error_maps[0], sal_maps[0])
    compare_two_images(error_maps[1], sal_maps[1])
    compare_two_images(error_maps[2], sal_maps[2])
    compare_two_images(error_maps[3], sal_maps[3])


def test_sal(fname, sname):
    imgs = load_array(fname)
    s_map = load_array(sname)
    print(imgs.shape)
    print(s_map.shape)


def test_sals(fname, sname, mname):
    imgs = load_array(fname)
    sal_maps = load_array(sname)
    model = load_model(mname)
    imgs = imgs[:, :, :, np.newaxis]
    train_size = int(math.ceil(len(imgs) * 85 / 100))
    ae_model = Autoencoder(imgs[:train_size], imgs[:train_size], imgs[train_size:], imgs[train_size:], model=model)
    ae_model.train(epochs=1)
    preds = ae_model.predict()
    plt.imshow(preds[0].squeeze())
    plt.show()
    error_maps = np.absolute(preds - imgs[train_size:])
    for i in range(6):
        img = error_maps[i].squeeze()
        plt.imshow(img)
        plt.show()


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

    run_trained_autoencoder(train_name, test_name, save_name, train=False)
    #test_sals(train_name, test_name, save_name)
    #run_sketch_autoencoder()



if __name__ == '__main__':
    main()