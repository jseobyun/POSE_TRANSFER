import os
import sys
import glob


def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

class Config:

    ## directory
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(cur_dir, '..')
    data_dir = os.path.join(root_dir, 'data')
    source_dir = os.path.join(data_dir, 'source')

    output_dir = os.path.join(root_dir, 'output')
    model_dir = os.path.join(output_dir, 'model_dump')
    vis_dir = os.path.join(output_dir, 'vis')
    pose_dir = os.path.join(root_dir, 'pose_detector')
    util_dir = os.path.join(root_dir, 'utils')

    ## input, output
    h = 288
    w = 512
    input_shape = (h, w)

    pixel_mean = (0.485, 0.456, 0.406)
    pixel_std = (0.229, 0.224, 0.225)

    ## training config
    continue_train = False

    lr_dec_epoch = [17, 21]
    start_epoch = 0
    if continue_train:
        model_file_list = glob.glob(os.path.join(model_dir, '*.pth.tar'))
        if len(model_file_list) !=0:
            last_epoch = max([int(file_name[file_name.find('G_') + 2: file_name.find('.pth.tar')]) for file_name in model_file_list])
            start_epoch = last_epoch
    num_epoch = 200
    lr = 1e-3
    lr_dec_factor = 10
    batch_size = 8
    test_batch_size = 1

    ## testing config
    flip_test = False  # True
    test_model_epoch = 20



    ## others
    sigma = 2.0
    num_thread = 5
    gpu_ids = '0,1,2,3'
    num_gpus = 1

    visualize = False
    do_augment = False

    num_stacks = 2
    num_blocks = 2


    pose_weight_path = os.path.join(pose_dir, 'weights/pose_model_old.pth')

    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

cfg = Config()
# sys.path.insert(0, os.path.join(cfg.root_dir, 'common'))
add_pypath(cfg.data_dir)
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)

