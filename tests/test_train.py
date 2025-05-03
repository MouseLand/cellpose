from cellpose import io, models, train
from subprocess import check_output, STDOUT
import os, shutil
import torch


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def test_class_train(data_dir):
    train_dir = str(data_dir.joinpath('2D').joinpath('train'))
    model_dir = str(data_dir.joinpath('2D').joinpath('train').joinpath('models'))
    shutil.rmtree(model_dir, ignore_errors=True)
    output = io.load_train_test_data(train_dir, mask_filter='_cyto_masks')
    images, labels, image_names, test_images, test_labels, image_names_test = output
    use_gpu = torch.cuda.is_available()
    model = models.CellposeModel(gpu=use_gpu)
    cpmodel_path = train.train_seg(model.net, images, labels, train_files=image_names,
                                   test_data=test_images, test_labels=test_labels,
                                   test_files=image_names_test,
                                   save_path=train_dir, n_epochs=3)[0]
    io.add_model(cpmodel_path)
    io.remove_model(cpmodel_path, delete=True)
    print('>>>> model trained and saved to %s' % cpmodel_path)


def test_cli_train(data_dir):
    # import sys
    # path_root = Path(__file__).parents[1]
    # sys.path.append(str(path_root))
    # print(Path(__file__).parents[0],Path(__file__).parents[1],Path(__file__).parents[2])
    train_dir = str(data_dir.joinpath('2D').joinpath('train'))
    model_dir = str(data_dir.joinpath('2D').joinpath('train').joinpath('models'))
    shutil.rmtree(model_dir, ignore_errors=True)
    use_gpu = torch.cuda.is_available()
    gpu_str = "--use_gpu" if use_gpu else ""
    cmd = 'python -m cellpose %s --train --n_epochs 3 --dir %s --mask_filter _cyto_masks --pretrained_model None' % (gpu_str, train_dir)
    try:
        cmd_stdout = check_output(cmd, stderr=STDOUT, shell=True).decode()
    except Exception as e:
        print(e)
        raise ValueError(e)

