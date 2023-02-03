# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from trainer import Trainer
from options import MonodepthOptions

options = MonodepthOptions()
opts = options.parse()
opts.no_cuda = True
opts.num_workers = 2
opts.png = True
opts.num_epochs = 1
opts.log_dir = '/Users/ilya/Документы/Учеба/Диплом/code/monodepth2/logs'
# opts.dataset = 'kitti_depth'
opts.log_frequency = 1

# структура файлов глубин
# depth_path = '/kitti_data/2011_09_26/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_03/0000000077.png'

if __name__ == "__main__":
    trainer = Trainer(opts)
    trainer.train()