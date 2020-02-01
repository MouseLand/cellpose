import numpy as np
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from . import utils
import cv2
from scipy.ndimage import gaussian_filter
import scipy
import skimage
from skimage import draw

def show_segmentation(fig, img, maski, flowi, save_path=None):
    #outpix = plot_outlines(maski)
    outlines = masks_to_outlines(maski)
    overlay = mask_overlay(img, maski)

    ax = fig.add_subplot(1,4,1)
    img0 = img.copy()
    if img0.shape[0] < 4:
        img0 = np.transpose(img0, (1,2,0))
    if img0.shape[-1] < 3:
        img0 = rgb_image(img0)
    else:
        if img0.max()<=50.0:
            img0 = np.uint8(np.clip(img0*255, 0, 1))
    ax.imshow(img0)
    ax.set_title('original image')
    ax.axis('off')

    ax = fig.add_subplot(1,4,2)
    outX, outY = np.nonzero(outlines)
    imgout= img0.copy()
    imgout[outX, outY] = np.array([255,75,75])
    ax.imshow(imgout)
    #for o in outpix:
    #    ax.plot(o[:,0], o[:,1], color=[1,0,0], lw=1)
    ax.set_title('predicted outlines')
    ax.axis('off')

    ax = fig.add_subplot(1,4,3)
    ax.imshow(overlay)
    ax.set_title('predicted masks')
    ax.axis('off')

    ax = fig.add_subplot(1,4,4)
    ax.imshow(flowi)
    ax.set_title('predicted cell pose')
    ax.axis('off')

    if save_path is not None:
        skimage.io.imsave(save_path + '_overlay.jpg', overlay)
        skimage.io.imsave(save_path + '_outlines.jpg', imgout)
        skimage.io.imsave(save_path + '_flows.jpg', flowi)



def outline_overlay(img, outlines, channels=[0,0], color=[255,0,0]):
    """ outlines are red by default """
    if not isinstance(outlines, list):
        outlines = [outlines]
        color = [color]
    img = np.clip(img.copy(), 0, 1)
    #for i in range(img.shape[-1]):
    #    img[:,:,i] = ((img[:,:,i]-img[:,:,i].min()) / 
    #                  (img[:,:,i].max()-img[:,:,i].min()))
    img *= 255
    img = np.uint8(img)
    RGB = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    if channels[0]==0:
        RGB = np.tile(img[:,:,0][:,:,np.newaxis],(1,1,3))
    else:
        RGB[:,:,channels[0]-1] = img[:,:,0]
        if channels[1] > 0:
            RGB[:,:,channels[1]-1] = img[:,:,1]
    for i in range(len(outlines)):
        RGB[outlines[i]>0] = np.array(color[i])
    return RGB

def mask_overlay(img, masks, colors=None):
    if colors is not None:
        if colors.max()>1:
            colors = np.float32(colors)
            colors /= 255
        colors = rgb_to_hsv(colors)
    img = utils.normalize99(img.astype(np.float32).mean(axis=-1))
    img -= img.min()
    img /= img.max()
    HSV = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    HSV[:,:,2] = np.clip(img*1.5, 0, 1.0)
    for n in range(int(masks.max())):
        ipix = (masks==n+1).nonzero()
        if colors is None:
            HSV[ipix[0],ipix[1],0] = np.random.rand()
        else:
            HSV[ipix[0],ipix[1],0] = colors[n,0]
        HSV[ipix[0],ipix[1],1] = 1.0
    RGB = (hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return RGB

def rgb_image(img):
    """ image is 2 x Ly x Lx - change to RGB Ly x Lx x 3 """
    if img.shape[-1]<3:
        img = np.transpose(img, (2,0,1))
    gray = False
    if np.ptp(img[1]) < 1e-3:
        gray = True
    img[0] = utils.normalize99(img[0])
    if not gray:
        img[1] = utils.normalize99(img[1])
    _,Ly,Lx = img.shape
    img = np.clip(img, 0., 1.)
    img = np.uint8(img*255)
    RGB = np.zeros((Ly, Lx, 3), np.uint8)
    if gray:
        RGB = np.tile(img[0][...,np.newaxis], (1,1,3))
    else:
        RGB[:,:,1] = img[0]
        RGB[:,:,2] = img[1]
    return RGB

def interesting_patch(mask, bsize=130):
    """ get patch of size bsize x bsize with most masks """
    Ly,Lx = mask.shape
    m = np.float32(mask>0)
    m = gaussian_filter(m, bsize/2)
    y,x = np.unravel_index(np.argmax(m), m.shape)
    ycent = max(bsize//2, min(y, Ly-bsize//2))
    xcent = max(bsize//2, min(x, Lx-bsize//2))
    patch = [np.arange(ycent-bsize//2, ycent+bsize//2, 1, int),
             np.arange(xcent-bsize//2, xcent+bsize//2, 1, int)]
    return patch

def disk(med, r, Ly, Lx):
    """ returns pixels of disk with radius r and center med """
    yy, xx = np.meshgrid(np.arange(0,Ly,1,int), np.arange(0,Lx,1,int),
                         indexing='ij')
    inds = ((yy-med[0])**2 + (xx-med[1])**2)**0.5 <= r
    y = yy[inds].flatten()
    x = xx[inds].flatten()
    return y,x

def circle(med, r):
    """ returns pixels of circle with radius r and center med """
    theta = np.linspace(0.0,2*np.pi,100)
    x = r * np.cos(theta) + med[0]
    y = r * np.sin(theta) + med[1]
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    return x,y

def dx_to_circ(dP):
    sc = max(np.percentile(dP[0], 99), np.percentile(dP[0], 1))
    Y = np.clip(dP[0] / sc, -1, 1)
    sc = max(np.percentile(dP[1], 99), np.percentile(dP[1], 1))
    X = np.clip(dP[1] / sc, -1, 1)
    #Y = (np.clip(utils.normalize99(dP[0]), 0,1) - 0.5) * 2
    #X = (np.clip(utils.normalize99(dP[1]), 0,1) - 0.5) * 2
    H = (np.arctan2(Y, X) + np.pi) / (2*np.pi)
    S = utils.normalize99(dP[0]**2 + dP[1]**2)
    V = np.ones_like(S)
    HSV = np.concatenate((H[:,:,np.newaxis], S[:,:,np.newaxis], S[:,:,np.newaxis]), axis=-1)
    HSV = np.clip(HSV, 0.0, 1.0)
    flow = (hsv_to_rgb(HSV)*255).astype(np.uint8)
    return flow

def masks_to_outlines(masks):
    Ly, Lx = masks.shape
    nmask = masks.max()
    outlines = np.zeros((Ly,Lx), np.bool)
    # pad T0 and mask by 2
    T = np.zeros((Ly+2)*(Lx+2), np.int32)
    Lx += 2
    iun = np.unique(masks)[1:]
    for iu in iun:
        y,x = np.nonzero(masks==iu)
        y+=1
        x+=1
        T[y*Lx + x] = 1
        T[y*Lx + x] =  (T[(y-1)*Lx + x]   + T[(y+1)*Lx + x] +
                        T[y*Lx + x-1]     + T[y*Lx + x+1] )
        outlines[y-1,x-1] = np.logical_and(T[y*Lx+x]>0 , T[y*Lx+x]<4)
        T[y*Lx + x] = 0
    #outlines *= masks
    return outlines

def masks_to_outlines_numbered(masks):
    Ly, Lx = masks.shape
    nmask = masks.max()
    outlines = np.zeros((Ly,Lx), np.int32)
    # pad T0 and mask by 2
    T = np.zeros((Ly+2)*(Lx+2), np.int32)
    Lx += 2
    for k in range(nmask):
        y,x = np.nonzero(masks==(k+1))
        y+=1
        x+=1
        T[y*Lx + x] = 1
        T[y*Lx + x] =  (T[(y-1)*Lx + x]   + T[(y+1)*Lx + x] +
                        T[y*Lx + x-1]     + T[y*Lx + x+1] )
        outlines[y-1,x-1] = (k+1)*np.logical_and(T[y*Lx+x]>0 , T[y*Lx+x]<4)
        T[y*Lx + x] = 0
    #outlines *= masks
    return outlines

def outline_from_mask(mask):
    Y = mask.copy()
    Ly, Lx = Y.shape
    mu = np.zeros((Ly,Lx))
    unq = np.unique(Y)
    nmask = len(unq)-1
    for j in range(nmask):
        mask = (Y==unq[j+1])
        y,x = (Y==unq[j+1]).nonzero()
        y0 = np.min(y)
        x0 = np.min(x)
        y = y-y0
        x = x-x0
        Ly, Lx = np.max(y)+1, np.max(x)+1
        ma0 = np.zeros((Ly,Lx))
        ma0[y,x] = 1
        ma = cv2.boxFilter(ma0, ksize=(3, 3), normalize=False, ddepth=-1)
        maskE = np.logical_and(ma < 9, ma0>.5)
        mu[y+y0,x+x0] = mu[y+y0,x+x0] +maskE[y,x]
    return mu


def plot_outlines(masks):
    outpix=[]
    for n in np.unique(masks)[1:]:
        mn = masks==n
        if mn.sum() > 0:
            contours, hierarchy = cv2.findContours(mn.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            #contours = measure.find_contours(mn, 0.5)
            cmax = np.argmax([c.shape[0] for c in contours])
            pix = contours[cmax].astype(int).squeeze()
            if len(pix)>4:
                pix=pix[:,::-1]
                pix = draw.polygon_perimeter(pix[:,0], pix[:,1], (mn.shape[0], mn.shape[1]))
                pix = np.array(pix).T[:,::-1]
                outpix.append(pix)
            else:
                outpix.append(np.zeros((0,2)))
    return outpix
