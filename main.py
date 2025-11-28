#!/usr/bin/env python
import copy
import torch
import random
import argparse
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time
import warnings
import numpy as np
from pathlib import Path
from model.MLModel import *
from model.myresnet import *
warnings.simplefilter("ignore")
from servers.serverFedDWA import FedDWA
from utils.logger import *
from utils.plot_utils import plot_training_results



def parse_args():
    parser = argparse.ArgumentParser()

    # general setting
    parser.add_argument('--device', type=str, default='gpu', choices=['gpu', 'cpu'])
    # parser.add_argument('--gpu', type=int, default=1, help='gpu id')
    parser.add_argument("--gpu", type=int, nargs='+', default=None, help="")
    parser.add_argument('--seed', type=int, default=12345, help='random seed')
    parser.add_argument('--num_classes', type=int, default=10, help='num_classes')
    parser.add_argument('--times', type=int, default=1, help='current time to run the algorithm')
    parser.add_argument('--dataset', type=str, default='cifar10tpds', help='dataset name',
                        choices=['cifar100tpds', 'cifar10tpds', 'cinic-10', 'tiny_ImageNet', 'gpr_custom'])
    parser.add_argument('--client_num', type=int, default=20, help='total client num')
    parser.add_argument('--client_frac', type=float, default=0.5, help='client fraction per round')
    parser.add_argument('--model', type=str, default='cnn', help='model type',
                        choices=['cnn', 'Resnet18',  'Resnet8', 'mobilevit', 'mobilevit_s'])
    parser.add_argument('--E', type=int, default=1, help='local epoch number per client')
    parser.add_argument('--Tg', type=int, default=100, help='global communication round')
    parser.add_argument('--B', type=int, default=20, help='client local batch size ')
    parser.add_argument('--lr', type=float, default=0.01, help='client local learning rate')
    parser.add_argument('--non_iidtype', type=int, default=1,
                        help="which type of non-iid is used, \
                             8 means pathological heterogeneous setting,\
                             10 means pracitical heterogeneous setting 1,\
                             9 means practical heterogeneous setting 2,", choices=[8, 9, 10,])
    parser.add_argument('--sample_rate', type=float, default=0.1, help="How much data to choose for training, range is (0,1]")
    parser.add_argument('--alpha_dir', type=float, default=0.1, help='hyper-parameter of dirichlet distribution')

    # [GPR-FedSense Arguments]
    parser.add_argument('--use_fedvls', action='store_true', help='Enable FedVLS (Vacant-class Distillation)')
    parser.add_argument('--fedvls_alpha', type=float, default=1.0, help='Weight for FedVLS distillation loss')
    parser.add_argument('--use_feddecorr', action='store_true', help='Enable FedDecorr (Feature Decorrelation)')
    parser.add_argument('--feddecorr_beta', type=float, default=0.1, help='Weight for FedDecorr loss')

    parser.add_argument('--data_dir', type=str, default='./data', help='root directory of the dataset')

    # dataset
    parser.add_argument('--num_types_noniid10', type=int, default=4,
                        help="The number of domain class for each client, range is [0,dataset classes], e.g.,for MNIST, [0,10]")
    parser.add_argument('--ratio_noniid10', type=float, default=0.8,
                        help='The radio of the domain class for each client, range is (0,1]')

    parser.add_argument('--alg', type=str, default='feddwa', help='algorithm',
                        choices=['feddwa'])


    # FedDWA
    parser.add_argument('--feddwa_topk', type=int, default=5,
                        help="hyper-parameter for feddwa (default=5)")
    parser.add_argument('--next_round', type=int, default=1,
                        help="hyper-parameter for feddwa (default=1)")


    return parser.parse_args()

def run_alg(args):

    # if args.device == 'gpu':
    #     args.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')
    if args.gpu == None:
        gpu_devices = '0'
    else:
        gpu_devices = ','.join([str(id) for id in args.gpu])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    if args.device == 'gpu':
        args.device = torch.device(f"cuda" if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    time_list = []
    print(f"\n============= Running time: {args.times}th =============")
    print("Creating server and clients ...")
    start = time.time()

    log_path = './logs_feddwa'
    os.makedirs(log_path, exist_ok=True)
    filename = f'{args.dataset}_{args.alg}_model={args.model}_C={args.client_frac}_osa={args.feddwa_topk}_next={args.next_round}_ratio={args.ratio_noniid10}_Tg={args.Tg}_N={args.client_num}_lr={args.lr}_E={args.E}_noniid={args.non_iidtype}_alpha={args.alpha_dir}_{args.seed}'
    log_path_name = os.path.join(log_path,filename)
    logger = LoggerCreator.create_logger(log_path = log_path_name, logging_name="Personalized FL", level=logging.INFO)
    logger.info(' '.join(f' \'{k}\': {v}, ' for k, v in vars(args).items()))

    # select model
    model_name = args.model
    modelObj = None
    if model_name == 'cnn':
        if args.dataset == 'cifar10tpds' or args.dataset == 'cinic-10':
            modelObj = CIFAR10Model(in_features=3, num_classes=10).to(args.device)
            args.num_classes = 10
        elif args.dataset == 'cifar100tpds':
            modelObj = CIFAR100Model(in_features=3, num_classes=100).to(args.device)
            args.num_classes = 100
        elif args.dataset == 'gpr_custom':
            modelObj = CIFAR10Model(in_features=3, num_classes=8).to(args.device)
            args.num_classes = 8
    elif model_name == 'Resnet8':
        if args.dataset == 'cifar10tpds':
            modelObj = Resnetwithoutcon_(option='resnet8',num_classes=10).to(args.device)
            args.num_classes = 10
        elif args.dataset == 'tiny_ImageNet':
            modelObj = Resnetwithoutcon_(option='resnet8',num_classes=200).to(args.device)
            args.num_classes = 200
        elif args.dataset == 'gpr_custom':
            modelObj = Resnetwithoutcon_(option='resnet8',num_classes=8).to(args.device)
            args.num_classes = 8
    elif model_name == 'Resnet18':
            if args.dataset == 'cifar10tpds':
                modelObj = Reswithoutcon(option='resnet18', num_classes=10).to(args.device)
                args.num_classes = 10
            elif args.dataset == 'tiny_ImageNet':
                modelObj = Reswithoutcon(option='resnet18',num_classes=200).to(args.device)
                args.num_classes = 200
            elif args.dataset == 'gpr_custom':
                modelObj = Reswithoutcon(option='resnet18',num_classes=8).to(args.device)
                args.num_classes = 8
    elif model_name == 'mobilevit' or model_name == 'mobilevit_s':
        if args.dataset == 'gpr_custom':
            # 使用 timm 的 mobilevit_s
            modelObj = MobileViT(model_name='mobilevit_s', num_classes=8).to(args.device)
            args.num_classes = 8
        else:
             # 默认 fallback
            modelObj = MobileViT(model_name='mobilevit_s', num_classes=args.num_classes).to(args.device)
    else:
        raise NotImplementedError

    # [Modified] Log model summary instead of full structure
    total_params = sum(p.numel() for p in modelObj.parameters())
    trainable_params = sum(p.numel() for p in modelObj.parameters() if p.requires_grad)
    logger.info(f"Model: {model_name}")
    logger.info(f"Total Parameters: {total_params:,}")
    logger.info(f"Trainable Parameters: {trainable_params:,}")

    # Save model structure to JSON
    import json
    model_structure = str(modelObj)
    structure_filename = f'{filename}_model_structure.txt' # Saving as txt might be more readable for structure string, but user asked for json
    structure_path = os.path.join(log_path, f"{filename}_model_structure.json")
    try:
        with open(structure_path, 'w') as f:
            json.dump({"model_structure": model_structure}, f, indent=4)
        logger.info(f"Model structure saved to: {structure_path}")
    except Exception as e:
        logger.error(f"Failed to save model structure: {e}")
    # select algorithm
    if args.alg == 'feddwa':
        server = FedDWA(args, modelObj, args.times,logger)
    else:
        raise NotImplementedError


    server.train()
    
    # [Added] Plot results automatically
    try:
        logger.info("Generating training plots...")
        plot_training_results()
        
        logger.info("Generating confusion matrices...")
        server.save_confusion_matrices()
    except Exception as e:
        logger.error(f"Failed to generate plots/matrices: {e}")
    time_list.append(time.time()-start)
    logger.info(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")


if __name__ == "__main__":
    args = parse_args()
    print(' '.join(f' \'{k}\': {v}, ' for k, v in vars(args).items()))
    run_alg(args)
