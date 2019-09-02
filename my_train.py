

import argparse

import os
import torch


# from mmdet import __version__

# from mmdet.apis import (get_root_logger, init_dist, set_random_seed,
#                         train_detector)
#
# from mmdet.datasets import build_dataset
# from mmdet.models import build_detector


from mmcv.utils import config
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, Runner, obj_from_dict



def parse_args():

    parser = argparse.ArgumentParser(description='Train a detector')
    # parser.add_argument('--config', type=str, default="E:/win10/InstanceSeg/mmdetection/configs/cityscapes/mask_rcnn_r50_fpn_1x_cityscapes.py", help='train config file path')
    parser.add_argument('--config', type=str, default="/home/huweiwei/InstanceSeg/mmdetection/configs/mask_rcnn_r50_fpn_1x_cityscapes.py", help='train config file path')
    parser.add_argument('--work_dir', type=str, default='/home/huweiwei/InstanceSeg/mmdetection/', help='the dir to save logs and models')

    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)


    return args


def main():

    # print("os.environ path are ", os.environ["PATH"])
    # print("os.pathsep are ", os.pathsep)
    # os.environ["PATH"] += os.pathsep + os.getcwd()

    args = parse_args()

    cfg = config.Config.fromfile(args.config)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir


    cfg.gpus = args.gpus

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8


    #init distributed env first, since logger depends on the dist info.
    # if args.launcher == 'none':
    #     distributed = False
    # else:
    #     distributed = True
    #     init_dist(args.launcher, **cfg.dist_params)
    #
    # # init logger before other steps
    # logger = get_root_logger(cfg.log_level)
    # logger.info('Distributed training: {}'.format(distributed))
    #
    # # set random seeds
    # if args.seed is not None:
    #     logger.info('Set random seed to {}'.format(args.seed))
    #     set_random_seed(args.seed)
    #
    # model = build_detector(
    #     cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)



if __name__ == '__main__':

    main()



