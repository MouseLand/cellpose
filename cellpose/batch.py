import numpy as np
import cv2


def batch_text(X, textures, xy = (224,224), scales = None, test = False):
    ni = len(X)
    imgi  = np.zeros((ni, 5, xy[0], xy[1]), 'float32')

    ym, xm = np.meshgrid(np.arange(xy[0]),np.arange(xy[1]))
    xymax = np.array(textures.shape[:2]) - np.array(xy)

    #veci  = np.zeros((ni, 2,   xy[0],   xy[1]), 'float32')
    #lbl   = np.zeros((ni, xy[0], xy[1]), 'int32')
    for k in range(ni):
        nchan, Ly, Lx = X[k][0].shape
        # generate random augmentations
        flip = np.random.rand()>.5
        theta = np.random.rand() * np.pi * 2
        if test is False:
            scale  = .75 + np.random.rand()/2
        else:
            scale = 1.
        scale *= 1./scales[k]

        dxy = np.maximum(0, np.array([Lx*scale-xy[1],Ly*scale-xy[0]]))
        dxy = (np.random.rand(2,) - .5) * dxy
        #print(theta, scale, dxy)
        cc = np.array([Lx/2, Ly/2])
        cc1 = cc - np.array([Lx-xy[1], Ly-xy[0]])/2 + dxy
        pts1 = np.float32([cc,cc + np.array([1,0]), cc + np.array([0,1])])
        pts2 = np.float32([cc1,
                cc1 + scale*np.array([np.cos(theta), np.sin(theta)]),
                cc1 + scale*np.array([np.cos(np.pi/2+theta), np.sin(np.pi/2+theta)])])
        M = cv2.getAffineTransform(pts1,pts2)

        img = X[k][0].copy()
        #img = np.concatenate((img, np.expand_dims(X[k][2], 0)), axis=0)
        if len(X[k][1])>0:
            img = np.concatenate((img, np.expand_dims(X[k][1], 0)), axis=0)

        nchan, Ly, Lx = img.shape
        if flip:
            img = img[:,:, ::-1]
            img[1] = - img[1]

        imgi[k,0,:,:]  = cv2.warpAffine(img[0],M,(xy[0],xy[1]), flags=cv2.INTER_LINEAR)
        if img.shape[0]>4:
            imgi[k,4,:,:]  = cv2.warpAffine(img[-1],M,(xy[0],xy[1]), flags=cv2.INTER_LINEAR)

        imgi[k,3,:,:]  = cv2.warpAffine(img[3],M,(xy[0],xy[1]), flags=cv2.INTER_NEAREST)

        # add texture augmentation here
        if test==False and np.random.rand()>.5:
            mask  = cv2.warpAffine(X[k][2], M,(xy[0],xy[1]), flags=cv2.INTER_NEAREST)
            r = (xymax * np.random.rand(2000, 2)).astype('int32')
            ir = int(np.random.rand()*textures.shape[-1])
            #mask = imgi[k,5].astype('int32')
            imgi[k,0,:,:] += 0.1 * textures[ym+ r[mask,0], xm+ r[mask,1], ir] * (mask>0.5)

        v1  = cv2.warpAffine(img[1],M,(xy[0],xy[1]), flags=cv2.INTER_LINEAR)
        v2  = cv2.warpAffine(img[2],M,(xy[0],xy[1]), flags=cv2.INTER_LINEAR)
        imgi[k,1,:,:] = (v1 * np.cos(-theta) + v2*np.sin(-theta))
        imgi[k,2,:,:] = (-v1 * np.sin(-theta) + v2*np.cos(-theta))


    lbl  = imgi[:,3,:,:] > .5
    veci = imgi[:,1:3,:,:]
    #mask = imgi[:,5,:,:]
    imgi = imgi[:,[0, 4],:,:]

    return imgi, veci, lbl#, mask

def withflow_twochan(X, xy = (224,224), aug=False, scales = None, test = False):
    ni = len(X)
    imgi  = np.zeros((ni, 5, xy[0], xy[1]), 'float32')
    #veci  = np.zeros((ni, 2,   xy[0],   xy[1]), 'float32')
    #lbl   = np.zeros((ni, xy[0], xy[1]), 'int32')
    for k in range(ni):
        nchan, Ly, Lx = X[k][0].shape
        # generate random augmentations
        flip = np.random.rand()>.5
        theta = np.random.rand() * np.pi * 2
        if test is False:
            scale  = .75 + np.random.rand()/2
        else:
            scale = 1.
        scale *= 1./scales[k]

        dxy = np.maximum(0, np.array([Lx*scale-xy[1],Ly*scale-xy[0]]))
        dxy = (np.random.rand(2,) - .5) * dxy
        #print(theta, scale, dxy)
        cc = np.array([Lx/2, Ly/2])
        cc1 = cc - np.array([Lx-xy[1], Ly-xy[0]])/2 + dxy
        pts1 = np.float32([cc,cc + np.array([1,0]), cc + np.array([0,1])])
        pts2 = np.float32([cc1,
                cc1 + scale*np.array([np.cos(theta), np.sin(theta)]),
                cc1 + scale*np.array([np.cos(np.pi/2+theta), np.sin(np.pi/2+theta)])])
        M = cv2.getAffineTransform(pts1,pts2)

        img = X[k][0][:].copy()
        flag = False
        if len(X[k][1])>0:
            flag = True
            img = np.concatenate((img, np.expand_dims(X[k][1], 0)), axis=0)
            #if np.random.rand()>.25 or test==True:

        nchan, Ly, Lx = img.shape
        if flip:
            img = img[:,:, ::-1]
            img[2] = - img[2]

        imgi[k,0,:,:]  = cv2.warpAffine(img[0],M,(xy[0],xy[1]), flags=cv2.INTER_LINEAR)
        if flag:
            imgi[k,-1,:,:]  = cv2.warpAffine(img[-1],M,(xy[0],xy[1]), flags=cv2.INTER_LINEAR)

        imgi[k,3,:,:]  = cv2.warpAffine(img[3],M,(xy[0],xy[1]), flags=cv2.INTER_NEAREST)
        #imgi[k,4,:,:]  = cv2.warpAffine(img[4],M,(xy[0],xy[1]), flags=cv2.INTER_LINEAR) / (scales[k] * 27.)

        v2  = cv2.warpAffine(img[1],M,(xy[0],xy[1]), flags=cv2.INTER_LINEAR)
        v1  = cv2.warpAffine(img[2],M,(xy[0],xy[1]), flags=cv2.INTER_LINEAR)
        imgi[k,2,:,:] = (v1 * np.cos(-theta) + v2*np.sin(-theta))
        imgi[k,1,:,:] = (-v1 * np.sin(-theta) + v2*np.cos(-theta))

    if aug:
        imgi = augment.elastic_transform(imgi, alpha=6, ngrid=16)
    #imgi2 = imgi

    lbl  = imgi[:,3,:,:] > .5
    #bou = lbl * (imgi[:,4,:,:] < .125)
    #bou = 1/ (1 + np.exp(-(imgi[:,4,:,:] - .5)/.25)) - 1/(1+np.exp(.5/.25))
    veci = imgi[:,1:3,:,:]
    imgi = imgi[:,[0, -1],:,:]

    return imgi, veci, lbl#, bou

def with_mask(X, xy = (224,224), aug=False, scales = None, test = False):
    ni = len(X)
    imgi  = np.zeros((ni, 2, xy[0], xy[1]), 'float32')
    lbli   = np.zeros((ni, xy[0], xy[1]), 'int32')
    for k in range(ni):
        nchan, Ly, Lx = X[k][0].shape
        # generate random augmentations
        flip = np.random.rand()>.5
        theta = np.random.rand() * np.pi * 2
        if test is False:
            scale  = .75 + np.random.rand()/2
        else:
            scale = 1.
        scale *= 1./scales[k]

        dxy = np.maximum(0, np.array([Lx*scale-xy[1],Ly*scale-xy[0]]))
        dxy = (np.random.rand(2,) - .5) * dxy
        #print(theta, scale, dxy)
        cc = np.array([Lx/2, Ly/2])
        cc1 = cc - np.array([Lx-xy[1], Ly-xy[0]])/2 + dxy
        pts1 = np.float32([cc,cc + np.array([1,0]), cc + np.array([0,1])])
        pts2 = np.float32([cc1,
                cc1 + scale*np.array([np.cos(theta), np.sin(theta)]),
                cc1 + scale*np.array([np.cos(np.pi/2+theta), np.sin(np.pi/2+theta)])])
        M = cv2.getAffineTransform(pts1,pts2)

        img = X[k][0][:1].copy()
        lbl = 1 + X[k][2].copy()
        flag = False
        if len(X[k][1])>0:
            flag = True
            img = np.concatenate((img, np.expand_dims(X[k][1], 0)), axis=0)

        nchan, Ly, Lx = img.shape
        if flip:
            img = img[:,:, ::-1]
            lbl = lbl[:,::-1]

        imgi[k,0,:,:]  = cv2.warpAffine(img[0],M,(xy[0],xy[1]), flags=cv2.INTER_LINEAR)
        if flag:
            imgi[k,-1,:,:]  = cv2.warpAffine(img[-1],M,(xy[0],xy[1]), flags=cv2.INTER_LINEAR)

        lbli[k,:,:]  = cv2.warpAffine(lbl,M,(xy[0],xy[1]), flags=cv2.INTER_NEAREST)

    return imgi,  lbli
