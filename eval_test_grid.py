from __future__ import absolute_import, division, print_function

import os
import cv2
import numpy as np
import warnings

from evaluate_depth import compute_errors, batch_post_process_disparity

warnings.filterwarnings('ignore', module='mmcv')
warnings.filterwarnings('ignore', module='torchvision')

import torch
from torch.utils.data import DataLoader

from layers import disp_to_depth
from utils import readlines
from options import MonodepthOptions
import datasets
import networks

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


class Tester:
    def __init__(self, opt):
        self.opt = opt

    def grid_search(self):
        print("-> Loading weights from {}".format(self.opt.load_weights_folder))
        print("\n  " + ("{:>8} | " * 8).format("epoch", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
        num_epochs = len(os.listdir(self.opt.load_weights_folder)) - 1  # exclude opt.json
        for epoch in range(num_epochs - 1, num_epochs - 12, -1):
            self.evaluate_epoch(epoch)
        print("\n-> Done!")


    def evaluate_epoch(self, epoch):
        """Evaluates a pretrained model using a specified test set
        """
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80
        opt = self.opt
        load_weights_folder = opt.load_weights_folder + f'weights_{epoch}'
        assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
            "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

        if opt.ext_disp_to_eval is None:

            filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))
            encoder_path = os.path.join(load_weights_folder, "encoder.pth")
            decoder_path = os.path.join(load_weights_folder, "depth.pth")

            encoder_dict = torch.load(encoder_path)

            img_ext = '.png' if opts.png else '.jpg'

            dataset = datasets.KITTIRAWDataset(opt.data_path, filenames,
                                               encoder_dict['height'], encoder_dict['width'],
                                               [0], 4, is_train=False, img_ext=img_ext)
            dataloader = DataLoader(dataset, 64, shuffle=False, num_workers=opt.num_workers,
                                    pin_memory=True, drop_last=False)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # encoder = networks.ResnetEncoder(opt.num_layers, False)
            encoder = networks.resnet_encoder.VAN_encoder(zero_layer_mlp_ratio=4, zero_layer_depths=2,  pretrained=False)
            # depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)
            depth_decoder = networks.depth_decoder.HRDepthDecoder(num_ch_enc=[64, 64, 128, 320, 512], use_super_res=True, convnext=False)
            # depth_decoder = networks.depth_decoder.VAN_decoder(mlp_ratios=(4, 4, 4, 4), depths=(2, 2, 3, 2))
            model_dict = encoder.state_dict()
            encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
            depth_decoder.load_state_dict(torch.load(decoder_path))

            encoder.to(device)
            depth_decoder.to(device)
            encoder.eval()
            depth_decoder.eval()

            pred_disps = []

            with torch.no_grad():
                for data in dataloader:
                    input_color = data[("color", 0, 0)].to(device)

                    if opt.post_process:
                        # Post-processed results require each image to have two forward passes
                        input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

                    output = depth_decoder(encoder(input_color))

                    pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
                    pred_disp = pred_disp.cpu()[:, 0].numpy()

                    if opt.post_process:
                        N = pred_disp.shape[0] // 2
                        pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])

                    pred_disps.append(pred_disp)

            pred_disps = np.concatenate(pred_disps)

        else:
            # Load predictions from file
            print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
            pred_disps = np.load(opt.ext_disp_to_eval)

            if opt.eval_eigen_to_benchmark:
                eigen_to_benchmark_ids = np.load(
                    os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

                pred_disps = pred_disps[eigen_to_benchmark_ids]

        if opt.save_pred_disps:
            output_path = os.path.join(
                load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
            print("-> Saving predicted disparities to ", output_path)
            np.save(output_path, pred_disps)

        if opt.no_eval:
            print("-> Evaluation disabled. Done.")
            quit()

        elif opt.eval_split == 'benchmark':
            save_dir = os.path.join(load_weights_folder, "benchmark_predictions")
            print("-> Saving out benchmark predictions to {}".format(save_dir))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for idx in range(len(pred_disps)):
                disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
                depth = STEREO_SCALE_FACTOR / disp_resized
                depth = np.clip(depth, 0, 80)
                depth = np.uint16(depth * 256)
                save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
                cv2.imwrite(save_path, depth)

            print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
            quit()

        gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
        gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)["data"]


        if opt.eval_stereo:
            print("   Stereo evaluation - "
                  "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
            opt.disable_median_scaling = True
            opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
        else:
            pass

        errors = []
        ratios = []

        for i in range(pred_disps.shape[0]):

            gt_depth = gt_depths[i]
            gt_height, gt_width = gt_depth.shape[:2]

            pred_disp = pred_disps[i]
            pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
            pred_depth = 1 / pred_disp

            if opt.eval_split == "eigen" or opt.eval_split == 'eigen_zhou':  # DISSECT
                mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

                crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                                 0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

            else:
                mask = gt_depth > 0

            pred_depth = pred_depth[mask]
            gt_depth = gt_depth[mask]

            pred_depth *= opt.pred_depth_scale_factor
            if not opt.disable_median_scaling:
                ratio = np.median(gt_depth) / np.median(pred_depth)
                ratios.append(ratio)
                pred_depth *= ratio

            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

            errors.append(compute_errors(gt_depth, pred_depth))

        if not opt.disable_median_scaling:
            ratios = np.array(ratios)
            med = np.median(ratios)

        mean_errors = np.array(errors).mean(0)
        mean_errors = [epoch] + mean_errors.tolist()

        print((" {: 8d}  " + "|{: 8.3f}  " * 7).format(*mean_errors) + "|")


if __name__ == "__main__":
    options = MonodepthOptions()
    opts = options.parse()
    opts.log_dir = 'logs'
    opts.eval_mono = True
    tester = Tester(opts)
    tester.grid_search()

