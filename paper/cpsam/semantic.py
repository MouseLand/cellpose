import numpy as np 
from scipy.stats import mode
from scipy import ndimage
from pathlib import Path
from natsort import natsorted
import tifffile
from cellpose import metrics, io
import fastremap
import torch
from torch import nn 
from tqdm import trange, tqdm
from cellpose import transforms, dynamics, vit_sam, models, train

cl_epithelial = np.array([255,0,0])
cl_lymphocyte = np.array([255,255,0])
cl_macrophage = np.array([0,255,0])
cl_neutrophil = np.array([0,0,255])
cl_colors = [cl_epithelial, cl_lymphocyte, cl_macrophage, cl_neutrophil]
cl_names = ["epithelial", "lymphocyte", "macrophage", "neutrophil"]
cl_colors = np.array(cl_colors)

def rgb_to_masks_classes(rgb):
    masks0 = np.zeros(rgb.shape[:2], "uint16")
    class0 = []
    class0 = np.zeros(rgb.shape[:2], "int") 
    j0 = 0
    for ic in range(4):
        c0 = ((rgb == cl_colors[ic][np.newaxis, np.newaxis, :]).sum(axis=-1) == 3)
        c0 = ndimage.label(c0)[0].astype("uint16")
        class0[c0 > 0] = ic+1
        masks0[c0 > 0] = c0[c0 > 0] + j0
        j0 += c0.max()
    return masks0, class0

    
def convert_monusac_train_data(root):
    img_files = [f for folder in natsorted(root.glob("*")) for f in natsorted(folder.glob("*.tif")) if "_masks" not in f.name]
    lbl_files = [Path(str(f).replace(".tif", ".xml")) for f in img_files]

    root0 = root / "semantic/"
    root0.mkdir(exist_ok=True, parents=True)

    from skimage import draw
    import xml.etree.ElementTree as ET
    from shapely.geometry import Polygon

    labels = ["Epithelial", "Lymphocyte", "Macrophage", "Neutrophil"]
    # Read xml file
    for i, (xml_file, img_file) in enumerate(zip(lbl_files, img_files)):
        img = io.imread(img_file)
        img_shape = img.shape[:2]
        tree = ET.parse(xml_file)
        root = tree.getroot()

        n_ary_masks = np.zeros((4, img_shape[0], img_shape[1]), "uint16")
        count = 0
        gt = 0

        #Generate n-ary mask for each cell-type                       
        for k in range(len(root)):
            #label = [x.attrib['Name'] for x in root[k][0]]
            #label = label[0]
            #label_id = labels.index(label)

            for child in root[k]:
                for x in child:
                    r = x.tag
                    if r == 'Attribute':
                        count = count+1
                        label = x.attrib['Name']
                        n_ary_mask = np.transpose(np.zeros(img_shape, "uint16")) 
                        try:
                            label_id = labels.index(label)
                        except:
                            print(label, label_id)
                        
                    if r == 'Region':
                        regions = []
                        vertices = x[1]
                        coords = np.zeros((len(vertices), 2))
                        for iv, vertex in enumerate(vertices):
                            coords[iv][0] = vertex.attrib['X']
                            coords[iv][1] = vertex.attrib['Y']        
                        regions.append(coords)
                        poly = Polygon(regions[0])  

                        vertex_row_coords = regions[0][:,0]
                        vertex_col_coords = regions[0][:,1]
                        fill_row_coords, fill_col_coords = draw.polygon(vertex_col_coords, vertex_row_coords, n_ary_masks.shape[1:])
                        gt = gt+1 #Keep track of giving unique valu to each instance in an image
                        n_ary_masks[label_id][fill_row_coords, fill_col_coords] = gt
        print(n_ary_masks.max(axis=(1,2)))
        masks = n_ary_masks.max(axis=0)
        
        psize = 512
        ny = max(1, img.shape[0] // psize)
        nx = max(1, img.shape[1] // psize)
        print(ny, nx)
        for j in range(ny):
            jend = img.shape[0] if j==ny-1 else (j+1) * psize
            for k in range(nx):
                kend = img.shape[1] if k==nx-1 else (k+1) * psize
                img0 = img[j * psize : jend, k * psize : kend]
                masks0 = masks[j * psize : jend, k * psize : kend].copy()
                n_ary_masks0 = n_ary_masks[:, j * psize : jend, k * psize : kend].copy()
                if (masks0>0).sum() > 0:
                    img0 = transforms.normalize_img(img0, axis=-1).transpose(2, 0, 1)[:3]
                    lbl_all = dynamics.labels_to_flows([masks0], device=torch.device("cuda"))[0]

                    cp_all = np.zeros(img0.shape[1:], "float32")
                    for c in range(n_ary_masks0.shape[0]):
                        cp_all[n_ary_masks0[c] > 0] = c+1
                        
                    lbls = np.concatenate((lbl_all[:1], cp_all[np.newaxis,...], lbl_all[1:]), axis=0)

                    tifffile.imwrite(root0 / "train" / f"monusac_{i}_{j}_{k}.tif", data=img0, compression="zlib")
                    tifffile.imwrite(root0 / "train" / f"monusac_{i}_{j}_{k}_flows.tif", data=lbls, compression="zlib")
                    tifffile.imwrite(root0 / "train" / f"monusac_{i}_{j}_{k}_masks.tif", data=fastremap.renumber(masks0.astype("uint16"))[0], compression="zlib")

def convert_monusac_test_data(root):
    """ test data from https://monusac-2020.grand-challenge.org/Data/ 
    
    conversion script adapted from https://github.com/ruchikaverma-iitg/MoNuSAC
    
    """
    import xml.etree.ElementTree as ET
    from shapely.geometry import Polygon
    from skimage import draw

    #root = Path("/media/carsen/ssd3/datasets_cellpose/images_HandE/MoNuSAC/MoNuSAC Testing Data and Annotations/")
    img_files = [f for folder in natsorted(root.glob("*")) if "semantic" not in str(folder) for f in natsorted(folder.glob("*.tif"))]
    img_files = natsorted(img_files)
    lbl_files = [Path(str(f).replace(".tif", ".xml")) for f in img_files]
    
    root0 = root / "semantic/"
    root0.mkdir(exist_ok=True, parents=True)

    labels = ["Epithelial", "Lymphocyte", "Macrophage", "Neutrophil", "Ambiguous"]
    # Read xml file
    for i, (xml_file, img_file) in tqdm(enumerate(zip(lbl_files, img_files))):
        folder = img_file.parent.name
        img = io.imread(img_file)
        img_shape = img.shape[:2]
        tree = ET.parse(xml_file)
        root = tree.getroot()

        n_ary_masks = np.zeros((5, img_shape[0], img_shape[1]), "uint16")
        count = 0
        gt = 0

        #Generate n-ary mask for each cell-type                       
        for k in range(len(root)):
            for child in root[k]:
                for x in child:
                    r = x.tag
                    if r == 'Attribute':
                        count = count+1
                        label = x.attrib['Name']
                        n_ary_mask = np.transpose(np.zeros(img_shape, "uint16")) 
                        try:
                            label_id = labels.index(label)
                        except:
                            print(label, label_id)
                            if "Ambiguous" in label:
                                label_id = 4
                        
                    if r == 'Region':
                        regions = []
                        vertices = x[1]
                        coords = np.zeros((len(vertices), 2))
                        if len(vertices) > 2:
                            for iv, vertex in enumerate(vertices):
                                coords[iv][0] = vertex.attrib['X']
                                coords[iv][1] = vertex.attrib['Y']        
                            regions.append(coords)
                            poly = Polygon(regions[0])  

                            vertex_row_coords = regions[0][:,0]
                            vertex_col_coords = regions[0][:,1]
                            fill_row_coords, fill_col_coords = draw.polygon(vertex_col_coords, vertex_row_coords, n_ary_masks.shape[1:])
                            gt = gt+1 #Keep track of giving unique valu to each instance in an image
                            n_ary_masks[label_id][fill_row_coords, fill_col_coords] = gt
        masks = n_ary_masks[:4].max(axis=0)
        ibad = n_ary_masks[-1]

        cp_all = np.zeros(img_shape, "float32")
        for j in range(4):
            cp_all[n_ary_masks[j] > 0] = j+1
            
        tifffile.imwrite(root0 / f"{img_file.stem}.tif", data=img, compression="zlib")
        tifffile.imwrite(root0 / f"{img_file.stem}_classes.tif", data=cp_all, compression="zlib")
        tifffile.imwrite(root0 / f"{img_file.stem}_masks.tif", data=masks.astype("uint16"), compression="zlib")
        tifffile.imwrite(root0 / f"{img_file.stem}_masks_bad.tif", data=ibad.astype("uint16"), compression="zlib")
        
def initialize_class_net(nclasses=5, device=torch.device("cuda")):
    net = vit_sam.Transformer(rdrop=0.4).to(device)
    # default model
    net.load_model("models/cpsam8_0_2100_8_402175188", device=device, strict=False, multigpu=False)
    
    # initialize weights for class maps
    ps = 8 # patch size
    nout = 3
    nclasses = 5
    w0 = net.out.weight.data.detach().clone()
    b0 = net.out.bias.data.detach().clone() 
    net.out = nn.Conv2d(256, (nout + nclasses) * ps**2, kernel_size=1).to(device)
    # set weights for background map
    i = 0
    net.out.weight.data[i * ps**2 : (i+1) * ps**2] = -0.5*w0[(nout-1) * ps**2 : nout * ps**2]
    net.out.bias.data[i * ps**2 : (i+1) * ps**2] = b0[(nout-1) * ps**2 : nout * ps**2]
    # set weights for maps to 4 nuclei classes
    for i in range(1, nclasses):
        net.out.weight.data[i * ps**2 : (i+1) * ps**2] = 0.5*w0[(nout-1) * ps**2 : nout * ps**2]
        net.out.bias.data[i * ps**2 : (i+1) * ps**2] = b0[(nout-1) * ps**2 : nout * ps**2]
    net.out.weight.data[-(nout * ps**2) : ] = w0 
    net.out.bias.data[-(nout * ps**2) : ] = b0
    net.W2 = nn.Parameter(torch.eye((nout + nclasses) * ps**2).reshape((nout + nclasses) * ps**2, nout + nclasses, ps, ps), 
                                requires_grad=False)
    net.to(device);
    return net

def train_net(root0):
    train_files = (root0 / "train").glob("*.tif")
    train_files = natsorted([tf for tf in train_files if "_flows" not in str(tf) and "_masks" not in str(tf)])
    train_data, test_data = [], []
    print("loading images")
    for i in trange(len(train_data)):
        img = io.imread(train_data[i])
        train_data.append(img)
            
    print("loading labels")
    train_labels = [io.imread(str(train_files[i])[:-4] + f'_flows.tif') for i in trange(len(train_files))]
    train_masks = [io.imread(str(train_files[i])[:-4] + f'_masks.tif') for i in trange(len(train_files))]
    
    pclass = np.zeros((5,))
    pclass_img = np.zeros((len(train_data), 5))
    for c in range(5):
        pclass_img[:, c] = np.array([(tl[1] == c).mean() for tl in train_labels])
    pclass = pclass_img.mean(axis=0)
    print(pclass)

    device = torch.device("cuda")
    
    net = initialize_class_net(nclasses=5, device=device)
    
    learning_rate = 5e-5 
    weight_decay = 0.1 
    batch_size = 16 
    n_epochs = 500
    bsize = 256
    rescale = False 
    scale_range = 0.5

    out = train.train_seg(net, train_data=train_data, train_labels=train_labels, 
                            learning_rate=learning_rate, weight_decay=weight_decay,
                            batch_size=batch_size, n_epochs=n_epochs,
                            bsize=bsize, 
                            nimg_per_epoch=len(train_data),
                            rescale=rescale, scale_range=scale_range,
                            min_train_masks=0,
                            nimg_test_per_epoch=len(test_data), 
                            model_name="he_monusac_pclass_final2_batch_size_16",
                            class_weights=1./pclass)
    
def test_net(root0):
    model = models.CellposeModel(gpu=True, nchan=3, pretrained_model=None)
    net = initialize_class_net(nclasses=5, device=torch.device("cuda"))
    net.load_model("models/he_monusac_pclass_final2_batch_size_16", device=torch.device("cuda"), strict=False, multigpu=False)
    net.eval()
    model.net = net
    model.net_ortho = None
    img_files = [f for f in natsorted(root0.glob("*tif")) if "_masks" not in str(f) and "_flows" not in str(f) and "_classes" not in str(f)]
    test_imgs = [io.imread(f)[:,:,:3] for f in img_files]
    masks_true = [io.imread(str(f).replace(".tif", "_masks.tif")).squeeze() for f in img_files]
    masks_bad = [io.imread(str(f).replace(".tif", "_masks_bad.tif")).squeeze() for f in img_files]
    classes_true = [io.imread(str(f).replace(".tif", "_classes.tif")) for f in img_files]

    masks_pred, flows, styles = model.eval(test_imgs, diameter=None, augment=False,
                                    bsize=256, tile_overlap=0.1, batch_size=64,
                                    flow_threshold=0.4, cellprob_threshold=0)
    
    classes_pred = [s.squeeze().argmax(axis=-1) for s in styles]
    
    aps_img, errors_img = compute_ap_pq(masks_true, masks_bad, classes_true, masks_pred, classes_pred)

    print(np.nanmean(errors_img, axis=0), np.nanmean(errors_img))
    print(np.nanmean(aps_img, axis=0), np.nanmean(aps_img))

    np.save("results/monusac_cellposeSAM.npy", {"errors": errors_img, "aps": aps_img, "masks_true": masks_true, "masks_pred": masks_pred, 
                                "classes_true": classes_true, "classes": classes_pred, 
                                "masks_bad": masks_bad, "imgs": test_imgs, "img_files": img_files})
    
def compute_leader_scores(root0):
    """ download leader masks from https://monusac-2020.grand-challenge.org/Data/ """
    
    img_files = [f for f in natsorted(root0.glob("*tif")) if "_masks" not in str(f) and "_flows" not in str(f) and "_classes" not in str(f)]
    masks_true = [io.imread(str(f).replace(".tif", "_masks.tif")).squeeze() for f in img_files]
    masks_bad = [io.imread(str(f).replace(".tif", "_masks_bad.tif")).squeeze() for f in img_files]
    classes_true = [io.imread(str(f).replace(".tif", "_classes.tif")) for f in img_files]

    leader_folders = ["Amirreza_Mahbod", "IIAI", "SJTU_426", "SharifHooshPardaz"]
    folders = natsorted(np.unique([f.name.split("_")[0] for f in img_files]))
    for lfolder in leader_folders:
        masks_pred_l = []
        classes_pred_l = []
        for folder in folders:
            ids = [i for i, f in enumerate(img_files) if folder in f.name] 
            iouc, tpc, fpc, fnc = np.zeros((len(ids), 4), "float32"), np.zeros((len(ids), 4), "int"), np.zeros((len(ids), 4), "int"), np.zeros((len(ids), 4), "int")
            for i in ids:
                rgb_file = Path(f"/media/carsen/ssd3/datasets_cellpose/images_HandE/MoNuSAC/{lfolder}/{folder}/{img_files[i].stem}_mask.png.tif")
                masks_pred0, classes_pred0 = rgb_to_masks_classes(io.imread(rgb_file))
                masks_pred_l.append(masks_pred0)
                classes_pred_l.append(classes_pred0)

        aps_img, errors_img = compute_ap_pq(masks_true, masks_bad, classes_true, masks_pred_l, classes_pred_l)
        print(np.nanmean(errors_img, axis=0), np.nanmean(errors_img))
        print(np.nanmean(aps_img, axis=0), np.nanmean(aps_img))

        np.save(f"results/monusac_{lfolder}.npy", {"errors": errors_img, "aps": aps_img, 
                                        "masks_pred": masks_pred_l, 
                                        "classes_pred": classes_pred_l})


def compute_ap_pq(masks_true, masks_bad, classes_true, masks_pred, classes_pred):
    nimg = len(masks_true)
    iou_all, tp_all, fp_all, fn_all = np.zeros((nimg, 4), "float32"), np.zeros((nimg, 4), "int"), np.zeros((nimg, 4), "int"), np.zeros((nimg, 4), "int")
    for i in range(nimg):
        masks_pred0 = masks_pred[i].copy() 
        masks_bad0 = masks_bad[i].copy()
        class_true = classes_true[i].copy()
        class0 = classes_pred[i].copy()
        masks_true0 = masks_true[i].copy()
        masks_true0 = fastremap.renumber(masks_true0)[0]

        # remove masks that overlap with non-labeled regions (masks_bad0)
        iout, inds = metrics.mask_ious(masks_bad0, masks_pred0)
        fastremap.mask(masks_pred0, inds[iout > 0.5], in_place=True)
        masks_pred0 = fastremap.renumber(masks_pred0)[0]

        # remove masks with class 0 (background)
        masks_pred0[class0 == 0] = 0
        masks_pred0 = fastremap.renumber(masks_pred0)[0]

        # class id for each mask is mode of class ids in mask
        cid = np.array([mode(class0[masks_pred0==j])[0] for j in range(1, masks_pred0.max()+1)])
        tid = np.array([mode(class_true[masks_true0==j])[0] for j in range(1, masks_true0.max()+1)])

        # match ground truth and predicted masks
        iout, inds = metrics.mask_ious(masks_true0, masks_pred0)
        inds[iout < 0.5] = 0 # keep matches > 0.5 IoU
        # class for matched masks
        cmatch = cid[[ind-1 for ind in inds if ind!=0]]
        # class for true masks that are matched
        tmatch = tid[inds!=0]
        inds_match = inds[inds!=0]
        for c in range(4):
            # true positive if predicted mask class matches true mask class
            tps = (cmatch == c+1) * (tmatch == c+1)
            iou_all[i, c] = (iout[inds!=0] * tps).sum() # scale by IoU
            tp_all[i, c] = tps.sum()
            # false negative for all missed masks with class == c+1
            fn_all[i, c] = (tid == c+1).sum() - tps.sum()
            # false positive if predicted mask class == c+1 and does not match true mask class
            not_tp = np.ones(masks_pred0.max(), "bool")
            not_tp[inds_match[tps]-1] = False
            fp_all[i, c] = (cid[not_tp] == c+1).sum() # ((cmatch == c+1) * (tmatch != c+1)).sum() +
        assert (fp_all[i].sum() + tp_all[i].sum()) == masks_pred0.max()
        assert (fn_all[i].sum() + tp_all[i].sum()) == masks_true0.max()
            
    aps_img = tp_all / (tp_all + fp_all + fn_all)
    errors_img = (fp_all + fn_all) / (tp_all + fn_all)
    
    errors_img[np.isinf(errors_img)] = np.nan
    return aps_img, errors_img