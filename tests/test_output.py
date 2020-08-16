from cellpose import io, models
from pathlib import Path
import subprocess
import os

def clear_output():
    for image_name in image_names:
        if '2D' in image_name:
            cached_file = str(data_dir_2D.joinpath(image_name).resolve())
        else:
            cached_file = str(data_dir_3D.joinpath(image_name).resolve())
        name, ext = os.path.splitext(cached_file)
        output = name + '_cp_masks' + ext
        if os.path.exists(output):
            os.remove(output)

def test_cli_2D(data_dir):
    process = subprocess.Popen('python -m cellpose --dir %s --chan 2 --chan2 3'%str(data_dir.join('2D').resolve()), 
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    for l in stdout:
        print(l)
    
    clear_output()

def check_output(runtype):
    """
    Helper function to check if outputs given by a test are exactly the same
    as the ground truth outputs.
    """
    for image_name in image_names:
    for i in range(nplanes):
        compare_list_of_outputs(i,
                                outputs_to_check,
                                get_list_of_test_data(outputs_to_check, test_data_dir, nplanes, nchannels, added_tag, i),
                                get_list_of_output_data(outputs_to_check, output_root, i)
        )


#def test_cli_3D(data_dir):
#    os.system('python -m cellpose --dir %s'%str(data_dir.join('3D').resolve()))

#def test_gray_2D(data_dir):
#    os.system('python -m cellpose ')
#    data_dir.join('2D').