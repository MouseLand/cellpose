from cellpose import io, models, metrics, plot
from pathlib import Path
from subprocess import check_output, STDOUT
import os, shutil

def test_class_train(data_dir, image_names):
    train_dir = str(data_dir.joinpath('2D').joinpath('train'))
    model_dir = str(data_dir.joinpath('2D').joinpath('train').joinpath('models'))
    shutil.rmtree(model_dir, ignore_errors=True)
    output = io.load_train_test_data(train_dir, mask_filter='_cyto_masks')
    images, labels, image_names, test_images, test_labels, image_names_test = output
    model = models.CellposeModel(pretrained_model=None, diam_mean=30)
    cpmodel_path = model.train(images, labels, train_files=image_names, 
                               test_data=test_images, test_labels=test_labels, test_files=image_names_test,
                               channels=[2,1], save_path=train_dir, n_epochs=10)
    print('>>>> model trained and saved to %s'%cpmodel_path)
        
def test_cli_train(data_dir, image_names):
    train_dir = str(data_dir.joinpath('2D').joinpath('train'))
    model_dir = str(data_dir.joinpath('2D').joinpath('train').joinpath('models'))
    shutil.rmtree(model_dir, ignore_errors=True)
    cmd = 'python -m cellpose --train --train_size --n_epochs 10 --dir %s --mask_filter _cyto_masks --pretrained_model cyto --chan 2 --chan2 1 --diameter 30'%train_dir
    try:
        cmd_stdout = check_output(cmd, stderr=STDOUT, shell=True).decode()
    except Exception as e:
        print(e) 
        raise ValueError(e)
