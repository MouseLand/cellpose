from cellpose import io, models, metrics, plot
from pathlib import Path
import subprocess, shlex
import os
import numpy as np
import matplotlib.pyplot as plt

r_tol, a_tol = 1e-2, 1e-2

def clear_output(data_dir, image_names):
    data_dir_2D = data_dir.joinpath('2D')
    data_dir_3D = data_dir.joinpath('2D')
    for image_name in image_names:
        if '2D' in image_name:
            cached_file = str(data_dir_2D.joinpath(image_name))
            ext = '.png'
        else:
            cached_file = str(data_dir_3D.joinpath(image_name))
            ext = '.tif'
        name, ext = os.path.splitext(cached_file)
        output = name + '_cp_masks' + ext
        if os.path.exists(output):
            os.remove(output)

def test_class_2D(data_dir, image_names):
    clear_output(data_dir, image_names)
    img = io.imread(str(data_dir.joinpath('2D').joinpath('rgb_2D.png')))
    model_types = ['cyto', 'nuclei']
    chan = [2,1]
    chan2 = [1,0]
    for m,model_type in enumerate(model_types):
        model = models.Cellpose(model_type=model_type)
        masks, flows, _, _ = model.eval(img, diameter=0, channels=[chan[m],chan2[m]])
        io.imsave(str(data_dir.joinpath('2D').joinpath('rgb_2D_cp_masks.png')), masks)
        check_output(data_dir, image_names, '2D', model_type)
        fig = plt.figure(figsize=(8,3))
        plot.show_segmentation(fig, img, masks, flows[0], channels=[chan[m],chan2[m]])

def test_class_3D(data_dir, image_names):
    clear_output(data_dir, image_names)
    model = models.Cellpose()
    img = io.imread(str(data_dir.joinpath('3D').joinpath('rgb_3D.tif')))
    model_types = ['cyto', 'nuclei']
    chan = [2,1]
    chan2 = [1,0]
    for m,model_type in enumerate(model_types):
        masks = model.eval(img, diameter=0, channels=[chan[m],chan2[m]])[0]
        io.imsave(str(data_dir.joinpath('3D').joinpath('rgb_3D_cp_masks.tif')), masks)
        check_output(data_dir, image_names, '3D', model_type)
        
def test_cli_2D(data_dir, image_names):
    clear_output(data_dir, image_names)
    model_types = ['cyto', 'nuclei']
    chan = [2,1]
    chan2 = [1,0]
    for m,model_type in enumerate(model_types):
        cmd = 'python -m cellpose --dir %s --pretrained_model %s --fast_mode --chan %d --chan2 %d --diameter 0 --save_png'%(str(data_dir.joinpath('2D')), model_type, chan[m], chan2[m])
        process = subprocess.Popen(cmd, 
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=True)
        stdout, stderr = process.communicate()
        print(stdout)
        print(stderr)
        check_output(data_dir, image_names, '2D', model_type)
        
def test_cli_3D(data_dir, image_names):
    clear_output(data_dir, image_names)
    model_types = ['cyto', 'nuclei']
    chan = [2,1]
    chan2 = [1,0]
    for m,model_type in enumerate(model_types):
        cmd = 'python -m cellpose --dir %s --do_3D --pretrained_model %s --fast_mode --chan %d --chan2 %d --diameter 25 --save_tif'%(str(data_dir.joinpath('3D')), model_type, chan[m], chan2[m])
        process = subprocess.Popen(cmd, 
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   shell=True)
        stdout, stderr = process.communicate()
        print(stdout)
        print(stderr)
        check_output(data_dir, image_names, '3D', model_type)
        
def check_output(data_dir, image_names, runtype, model_type):
    """
    Helper function to check if outputs given by a test are exactly the same
    as the ground truth outputs.
    """
    data_dir_2D = data_dir.joinpath('2D')
    data_dir_3D = data_dir.joinpath('3D')
    for image_name in image_names:
        if '2D' in runtype and '2D' in image_name:
            image_file = str(data_dir_2D.joinpath(image_name))
            name, ext = os.path.splitext(image_file)
            output_test = name + '_cp_masks.png'
            output_true = name + '_%s_masks.png'%model_type
            check = True
        elif '3D' in runtype and '3D' in image_name:
            image_file = str(data_dir_3D.joinpath(image_name))
            name, ext = os.path.splitext(image_file)
            output_test = name + '_cp_masks.tif'
            output_true = name + '_%s_masks.tif'%model_type
            check = True

        if check:
            if os.path.exists(output_test):
                print('checking output %s'%output_test)
                masks_test = io.imread(output_test)
                masks_true = io.imread(output_true)
                ap = metrics.average_precision(masks_true, masks_test)
                print('average precision of [%0.3f %0.3f %0.3f]'%(ap[0],ap[1],ap[2]))
                yield np.allclose(ap, np.ones(3), rtol=r_tol, atol=a_tol)

                matching_pix = np.logical_and(masks_test>0, masks_true>0).mean()
                all_pix = (masks_test>0).mean()
                yield np.allclose(all_pix, matching_pix, rtol=r_tol, atol=a_tol)
            else:
                print('ERROR: no file of name %s found'%output_test)
                assert False
    clear_output(data_dir, image_names)

#def test_cli_3D(data_dir):
#    os.system('python -m cellpose --dir %s'%str(data_dir.join('3D').resolve()))

#def test_gray_2D(data_dir):
#    os.system('python -m cellpose ')
#    data_dir.join('2D').