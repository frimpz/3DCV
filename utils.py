import progressbar


class MyProgressBar():
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


# Download required files and create the needed directories
def setup():
    dirs = ['input', 'models', 'output', 'output/semantic', 'output/panoptic', 'output/instance']
    from pathlib import Path
    import os

    for i in dirs:
        Path(i).mkdir(parents=True, exist_ok=True)

    # check if model exist

    import urllib.request
    models = [(
              'https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
              'models/inception_semantic.h5', 'semantic segmentation'),
              ('https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.2/mask_rcnn_coco.h5',
               'models/mask_r_cnninstance.h5', 'instance segmentation'), ]
    for i in models:
        if not os.path.isfile(i[1]):
            print('Downloading model for '+ i[2] + ' >>>>>>>>>>')
            urllib.request.urlretrieve(i[0], i[1], MyProgressBar())


setup()
