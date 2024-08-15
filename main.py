import time
import random
import torch
import logging
import argparse
import os
import numpy as np
import warnings
import torch.multiprocessing
from train import train

torch.cuda.current_device()
torch.multiprocessing.set_sharing_strategy('file_system')
localtime = time.localtime(time.time())
str_time = f'{str(localtime.tm_mon)}-{str(localtime.tm_mday)}-{str(localtime.tm_hour)}-{str(localtime.tm_min)}'

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
        
def set_log(args):
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    log_file_path = os.path.join(args.logs_dir, f'{args.dataset}-{args.name}-{str_time}.log')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    for ph in logger.handlers:
        logger.removeHandler(ph)
    formatter_file = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter_file)
    logger.addHandler(fh)
    formatter_stream = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter_stream)
    logger.addHandler(ch)
    return logger


if __name__ == '__main__':
    start = time.time()
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='CMAWSC', help='project name')
    parser.add_argument('--dataset', type=str, default='Weibo17', help='support Weibo17/Weibo21')
    parser.add_argument('--method', type=str, default='CMAWSC', help='support CMAWSC')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
    parser.add_argument('--num_epoch', type=int, default=20, help='num_epoch')
    parser.add_argument('--num_workers',type=int, default= 10, help= 'num_workers')
    parser.add_argument('--text_encoder', type=str, default='bert-base-chinese', help='bert-base-chinese')
    parser.add_argument('--image_encoder', type=str, default='vit-base', help='vit-base')
    parser.add_argument('--lr_mm', type=float, default=1e-3, help='--lr_mm')
    parser.add_argument('--lr_mm_cls', type=float, default=1e-3, help='--lr_mm_cls')
    parser.add_argument('--weight_decay_tfm', type=float, default=1e-3, help='--weight_decay_tfm')
    parser.add_argument('--weight_decay_other', type=float, default=1e-2, help='--weight_decay_other')
    parser.add_argument('--lr_patience', type=float, default=3, help='--lr_patience')
    parser.add_argument('--lr_factor', type=float, default=0.2, help='--lr_factor')
    parser.add_argument('--lr_text_tfm', type=float, default=2e-5, help='--lr_text_tfm')
    parser.add_argument('--lr_img_tfm', type=float, default=5e-5, help='--lr_img_tfm')
    parser.add_argument('--lr_img_cls', type=float, default=1e-4, help='--lr_img_cls')
    parser.add_argument('--lr_text_cls', type=float, default=5e-5, help='--lr_text_cls')
    parser.add_argument('--data_dir', type=str, default='datasets', help='data_dir')
    parser.add_argument('--test_only', type=bool, default=False, help='train+test or test only')
    parser.add_argument('--pretrained_dir', type=str, default='Pretrained', help='path to pretrained models from Hugging Face.')
    parser.add_argument('--model_save_dir', type=str, default='results/models', help='path to save model parameters.')
    parser.add_argument('--res_save_dir', type=str, default='results/results', help='path to save training results.')
    parser.add_argument('--logs_dir', type=str, default='results/logs', help='path to log results.') 
    parser.add_argument('--seed', nargs='+', default=1, help='List of random seeds')

    args = parser.parse_args()
    logger = set_log(args)
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    args.device = device

    logger.info("Pytorch version: " + torch.__version__)
    logger.info("CUDA version: " + torch.version.cuda)
    logger.info(f"CUDA device: + {torch.cuda.current_device()}")
    logger.info(f"CUDNN version: {torch.backends.cudnn.version()}")
    logger.info("GPU name: " + torch.cuda.get_device_name())
    logger.info("Current Hyper-Parameters:")
    logger.info(args)

    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    name_seed = args.name + '_' + str(args.seed)
    args.model_save_path = os.path.join(args.model_save_dir, f'{args.dataset}-{name_seed}-{str_time}.pth')
    args.best_model_save_path = os.path.join(args.model_save_dir, f'{args.dataset}-{name_seed}-best.pth')

    setup_seed(args.seed)
    if args.dataset in ['Weibo17','Weibo21']:
        train(args)
    else:
        logger.info('数据集无效')
    end = time.time()
    logger.info(f"Run {end - start} seconds in total！")
