import os 
import sys
import json
import torch
import random
import logging
import argparse
import warnings
import numpy as np
from tqdm.auto import tqdm
import torch.optim as optim 
from torch.autograd import Variable
from torch.utils.data import dataloader,random_split
from thop import profile
# from tensorboardX import SummaryWriter
import time
import clip 
import utils
import datasets
import test_BLIP2 as test
import math  
from itertools import product 
from data_utils import squarepad_transform, targetpad_transform
from torch.cuda.amp import autocast as autocast, GradScaler

import torch.distributed as dist
import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta
from torch.utils.data.distributed import DistributedSampler
import setproctitle
from lavis.models import load_model_and_preprocess
from lavis.models.Llm_online.gpt import GPT4o
from torch.optim.lr_scheduler import LambdaLR

from cirr_sub_BLIP2 import test_cirr_submit_result

proc_title = "python-c"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
setproctitle.setproctitle(proc_title)
warnings.filterwarnings("ignore")
torch.set_num_threads(2)

parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', -1), type=int)
parser.add_argument('--dataset', default = '', help = "data set type")
parser.add_argument('--fashioniq_path', default = " ")
parser.add_argument('--shoes_path', default = "")
parser.add_argument('--cirr_path', default = "") 
parser.add_argument('--birds_path', default = "")
parser.add_argument('--Fashion200k_path', default = "")
parser.add_argument('--lasco_path', default = "")

parser.add_argument('--optimizer', default = 'adamw')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--EKI_lr', type=float, default=5e-4)
parser.add_argument('--noise_ratio', type=float, default=0.2,help='noise_ratio')

parser.add_argument('--device',type=str , default='cuda:0')

parser.add_argument('--model_dir', default='./test',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--save_summary_steps', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--node', type=str, default='')
parser.add_argument('--EKI_batch', default=25, type=int)
parser.add_argument('--recon_w',default=0.6,type =float)
parser.add_argument('--mc_drop',default=0.10,type =float)
parser.add_argument('--alpha',default=0.7,type =float)
parser.add_argument('--warmup', default=5, type=int)


args = parser.parse_args()

if args.dataset == "fashion200k":
    torch.multiprocessing.set_sharing_strategy('file_system')
def load_EKI_dataset():
    target_ratio = 1.25
    input_dim = 224
    preprocess = targetpad_transform(target_ratio, input_dim)
    img_transform = preprocess
    if args.dataset == 'fashioniq':
        EKI_set = datasets.FashionIQ_V(
            path = args.fashioniq_path,
            transform = img_transform,
            noise_ratio=args.noise_ratio
        )
    elif args.dataset == 'cirr':
        EKI_set = datasets.CIRR_V(
            path = args.cirr_path,
            transform = img_transform,
            case_look=False,
            noise_ratio=args.noise_ratio
        )
    train_ratio = args.EKI_batch / 100
    train_size = int(train_ratio * len(EKI_set))
    val_size = len(EKI_set) - train_size
    generator = torch.Generator()
    EKI_set, _ = random_split(EKI_set, [train_size, val_size], generator=generator)
    return EKI_set
def load_dataset():
    """Loads the input datasets."""
    print('Reading dataset ', args.dataset)
    transform = "targetpad"
    input_dim = 224
    target_ratio = 1.25
    if transform == "squarepad":
        preprocess = squarepad_transform(input_dim)
        print('Square pad preprocess pipeline is used')
    elif transform == "targetpad":
        #target_ratio = kwargs['target_ratio']
        preprocess = targetpad_transform(target_ratio, input_dim)
        print(f'Target pad with {target_ratio = } preprocess pipeline is used')
    else:
        raise ValueError("Preprocess transform should be in ['clip', 'squarepad', 'targetpad']")
    img_transform = preprocess

    if args.dataset == 'fashioniq':
        trainset = datasets.FashionIQ(
            path = args.fashioniq_path,
            transform=img_transform,
            noise_ratio=args.noise_ratio)
        trainset.shuffle()
        return trainset
    elif args.dataset == 'shoes':
        trainset = datasets.Shoes(
            path = args.shoes_path,
            transform=img_transform)
    elif args.dataset == 'cirr':
        trainset = datasets.CIRR(
            path = args.cirr_path,
            transform = img_transform,
            case_look=False,
            noise_ratio=args.noise_ratio
        )
        trainset.shuffle()
        return trainset
    elif args.dataset == 'lasco':
        trainset = datasets.LaSCo(
            path = args.lasco_path,
            transform = img_transform,
            case_look=False
        )
    elif args.dataset == 'birds':
        trainset = datasets.Birds(
            path = args.birds_path,
            transform = img_transform,
            split = 'train'
        )
        testset = datasets.Birds(
            path = args.birds_path,
            transform = img_transform,
            split = 'test'
        )
        print('trainset size:', len(trainset))
        print('test size:', len(testset))
        
        return trainset, testset
    elif args.dataset == 'fashion200k':
        trainset = datasets.Fashion200k(
            path = args.Fashion200k_path,
            transform = img_transform,
            split = 'train'
        )
        testset = datasets.Fashion200k(
            path = args.Fashion200k_path,
            transform = img_transform,
            split = 'test'
        )
        print('trainset size:', len(trainset))
        print('test size:', len(testset))
        return trainset, testset
    else:
        print('Invalid dataset', args.dataset)
        sys.exit()

    print('trainset size:', len(trainset))

    return trainset

def set_bn_eval(m): 
    classname = m.__class__.__name__ 
    if classname.find('BatchNorm2d') != -1: 
        m.eval()
    
def create_model_and_optimizer():
    blip_model_name = "Blip2QformerCir"
    backbone = "pretrain"
    model, vis_processors, txt_processors = load_model_and_preprocess(name=blip_model_name, model_type=backbone, is_eval=False, device=args.device)
    
    model.alpha = args.alpha
    model.cuda()
    eki_params = [p for n, p in model.named_parameters() if 'eki' in n and p.requires_grad]
    base_params = [p for n, p in model.named_parameters() if 'eki' not in n and p.requires_grad]

    optimizer = optim.AdamW([
        {'params': base_params, 'lr': args.lr},       
        {'params': eki_params, 'lr': args.EKI_lr}     
    ], weight_decay=0.05)

    return model, optimizer, txt_processors
    


def train(model, optimizer, dataloader,eki ,scaler, epoch, txt_processors):

    model.train()
    model.apply(set_bn_eval)
    summ = []
    loss_avg = utils.RunningAverage()
    with tqdm(total = len(dataloader)) as t:
        for i, data in enumerate(dataloader):
            if args.dataset == 'fashion200k':
                assert type(data) is list
                img1 = np.stack([d['source_img_data'] for d in data])
                img1 = torch.from_numpy(img1).float()
                img1 = torch.autograd.Variable(img1).cuda()
                img2 = np.stack([d['target_img_data'] for d in data])
                img2 = torch.from_numpy(img2).float()
                img2 = torch.autograd.Variable(img2).cuda()
                mods = [str(d['mod']['str']) for d in data]
                mods = [t.encode('utf-8').decode('utf-8') for t in mods]
            else:
                img1 = data['source_img_data'].cuda()
                img2 = data['target_img_data'].cuda()
                mods = data['mod']['str']
            captions = [txt_processors["eval"](caption) for caption in mods]
            optimizer.zero_grad()
            
            with autocast():
                samples={
                        "image":img1,
                        "target":img2,
                        "text_input":captions,
                        "labels":data['Qlabels'] if eki else None,
                        "eki":eki,
                        }
                loss_dict = model(samples,args.device)
                total_loss = 0.
                if(eki):
                    total_loss=loss_dict['loss_eki']
                else:
                    total_loss=loss_dict['loss_align'] + args.recon_w * loss_dict['loss_recon']
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            
            if i % args.save_summary_steps == 0:
                summary_batch = {}
                summary_batch['total_loss'] = total_loss.item()
                summ.append(summary_batch)
            loss_avg.update(total_loss.item())
            if eki:
                t.set_postfix(loss_EKI='{:05.3f}'.format(loss_avg()),lr = optimizer.param_groups[1]['lr'])
            else:
                t.set_postfix(loss_align='{:05.3f}'.format(loss_avg()),loss_recon = '{:05.3f}'.format(loss_dict['loss_recon']),lr = optimizer.param_groups[0]['lr'])
            t.update(1)


def train_and_evaluate(model, optimizer, trainset, EKI_set, txt_processors):

    c_trainloader = dataloader.DataLoader(trainset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=args.num_workers)
    v_trainloader = dataloader.DataLoader(EKI_set,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    drop_last=True,
                                    num_workers=args.num_workers) if args.EKI_batch>0 else []
    current_best_score = float('-inf')
    best_parameters_model = None
    test_metrics = {}
    scaler = GradScaler()
    epoches = args.num_epochs
    tolerance = 0

    for epoch in range(epoches):
        eki = True if epoch<args.warmup else False
        logging.info("Epoch {}/{}".format(epoch, epoches))

        trainloader = v_trainloader if eki else c_trainloader
        train(model, optimizer, trainloader, eki ,scaler, epoch, txt_processors)
        current_score = 0
        current_result = []
        if not eki:
            if args.dataset == 'fashioniq':
                for ci, category in enumerate(['dress', 'shirt', 'toptee']):
                    t = test.test(args, model, trainset, category, txt_processors)
                    logging.info(t)
                    current_score = current_score + t[1][1]
                    current_result.append(t)

                torch.save(model, os.path.join(args.model_dir, f'model_epoch_{epoch}.pt'))
                if current_score > current_best_score:
                    current_best_score = current_score
                    tolerance = 0
                    best_json_path_combine = os.path.join(
                            args.model_dir, "metrics_best.json")
                    test_metrics = {}

                    for _ in current_result:
                        for metric_name, metric_value in _:
                            test_metrics[metric_name] = metric_value

                    utils.save_dict_to_json(test_metrics, best_json_path_combine)
                    best_parameters_model = model
            else:
                if  args.dataset == 'cirr':
                    torch.save(model, os.path.join(args.model_dir, f'model_epoch_{epoch}.pt'))
                    t = test.test_cirr_valset(args, model, trainset, txt_processors)
                    logging.info(t)
                    current_score = t[0][1] + t[1][1] + t[2][1] + t[3][1] + t[4][1] + t[5][1] + t[6][1] # mean best
                    test_cirr_submit_result(model,save_dir = args.model_dir,testset= trainset,batch_size=128,name=f'_epoch_{epoch}',txt_processors=txt_processors)

                if current_score > current_best_score:
                    current_best_score = current_score
                    tolerance = 0
                    best_json_path_combine = os.path.join(
                            args.model_dir, "metrics_best.json")
                    test_metrics = {}
                    for metric_name, metric_value in t:
                        test_metrics[metric_name] = metric_value
                    torch.save(model, os.path.join(args.model_dir, f'model_epoch_{epoch}.pt'))
                    utils.save_dict_to_json(test_metrics, best_json_path_combine)
                    best_parameters_model = model 
        
    return current_best_score, test_metrics, best_parameters_model

if __name__ == '__main__':
    print("Here")
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    # Load the parameters from json file
    proc_title = "python-c"
    setproctitle.setproctitle(proc_title)
    print('Arguments:')
    for k in args.__dict__.keys():
        info = '    '+k+':'+str(args.__dict__[k])
        logging.info(info)

    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info('Loading the datasets and model...')

    trainset = load_dataset()

    best_score = float('-inf')

    EKI_set = load_EKI_dataset()
    model, optimizer, txt_processors = create_model_and_optimizer()


    logging.info("Starting train for {} epoch(s)".format(args.num_epochs))
    _best_score, _metrics, current_model = train_and_evaluate(model, optimizer, trainset, EKI_set, txt_processors)
    if _best_score > best_score:
        best_score = _best_score
        utils.save_dict_to_json(_metrics, os.path.join(args.model_dir, "metrics_best.json"))
        torch.save(current_model, os.path.join(args.model_dir, 'best_model.pt'))