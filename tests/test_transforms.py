from cellpose.transforms import *
from cellpose import io


def test_random_rotate_and_resize__default():
    nimg = 2
    X = [np.random.rand(64, 64) for i in range(nimg)]

    random_rotate_and_resize(X)


def test_normalize_img(data_dir):
    img = io.imread(str(data_dir.joinpath('3D').joinpath('rgb_3D.tif')))
    img = img.transpose(0, 2, 3, 1).astype('float32')

    img_norm = normalize_img(img, norm3D=True)
    assert img_norm.shape == img.shape

    img_norm = normalize_img(img, norm3D=True, tile_norm_blocksize=25)
    assert img_norm.shape == img.shape

    img_norm = normalize_img(img, norm3D=False, sharpen_radius=8)
    assert img_norm.shape == img.shape

def test_resize(data_dir):
    img = io.imread(str(data_dir.joinpath('2D').joinpath('rgb_2D_tif.tif')))
    
    Lx = 100
    Ly = 200
    
    img8 = resize_image(img.astype("uint8"), Lx=Lx, Ly=Ly)
    assert img8.shape == (Ly, Lx, 3)
    assert img8.dtype == np.uint8
    
    img16 = resize_image(img.astype("uint16"), Lx=Lx, Ly=Ly)
    assert img16.shape == (Ly, Lx, 3)
    assert img16.dtype == np.uint16
    
    img32 = resize_image(img.astype("uint32"), Lx=Lx, Ly=Ly)
    assert img32.shape == (Ly, Lx, 3)
    assert img32.dtype == np.uint32
    
