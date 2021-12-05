from cv2 import normalize
import numpy as np

from cellpose.transforms import *

def test_random_rotate_and_resize__default():
    nimg = 2
    X = [np.random.rand(64, 64) for i in range(nimg)]

    random_rotate_and_resize(X)

def test_random_rotate_and_resize__use_skel():
    nimg = 2
    X = [np.random.rand(64, 64) for i in range(nimg)]

    random_rotate_and_resize(X, omni=True)

def test_reshape_train_test__empty_labels():
    img_size = (16, 16)
    channels = (0, 0)
    high_val = (2**16)-1
    normalize = True
    
    train_data = [np.random.randint(0, high_val, size=img_size)]
    train_labels = [np.zeros(img_size, dtype=int)]

    test_data, test_labels = train_data, train_labels

    reshape_train_test(
        train_data, train_labels,
        test_data, test_labels,
        channels, normalize)