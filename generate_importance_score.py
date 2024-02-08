import torch
import numpy as np
import torchvision
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os, sys
import argparse
import pickle

from core.model_generator import wideresnet, preact_resnet, resnet
from core.training import Trainer, TrainingDynamicsLogger
from core.data import IndexDataset, CIFARDataset, SVHNDataset, CINIC10Dataset
from core.utils import print_training_info, StdRedirect

# med
from torchvision.models import resnet18, resnet50                   
import medmnist
from medmnist import INFO, Evaluator
import torch.utils.data as data

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

######################### Data Setting #########################
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='input batch size for training.')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'tiny', 'svhn', 'cinic10', 'organamnist', 'organsmnist'])
parser.add_argument('--download', action="store_true")  # med
parser.add_argument('--as_rgb', help='convert the grayscale image to RGB', action="store_true")  # med

######################### Path Setting #########################
parser.add_argument('--data_dir', type=str, default='../data/',
                    help='The dir path of the data.')
parser.add_argument('--base_dir', type=str,
                    help='The base dir of this project.')
parser.add_argument('--task_name', type=str, default='tmp',
                    help='The name of the training task.')

######################### GPU Setting #########################
parser.add_argument('--gpuid', type=str, default='0',
                    help='The ID of GPU.')

args = parser.parse_args()

######################### Set path variable #########################
task_dir = os.path.join(args.base_dir, args.task_name)
ckpt_path = os.path.join(task_dir, f'ckpt-last.pt')
td_path = os.path.join(task_dir, f'td-{args.task_name}.pickle')
data_score_path = os.path.join(task_dir, f'data-score-{args.task_name}.pickle')

######################### Print setting #########################
print_training_info(args, all=True)

#########################
dataset = args.dataset
if dataset in ['cifar10', 'svhn', 'cinic10']:
    num_classes=10
elif dataset == 'cifar100':
    num_classes=100
else:
    info = INFO[args.dataset]
    download = args.download
    as_rgb = args.as_rgb
    num_classes = len(info['label'])


######################### Ftn definition #########################
"""Calculate loss and entropy"""
def post_training_metrics(model, dataloader, data_importance, device):
    model.eval()
    data_importance['entropy'] = torch.zeros(len(dataloader.dataset))
    data_importance['loss'] = torch.zeros(len(dataloader.dataset))

    for batch_idx, (idx, (inputs, targets)) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        targets = targets.squeeze()  # if MedMNIST

        logits = model(inputs)
        prob = nn.Softmax(dim=1)(logits)

        entropy = -1 * prob * torch.log(prob + 1e-10)
        entropy = torch.sum(entropy, dim=1).detach().cpu()
        # breakpoint()
        # print(f'logits: {logits.shape}, targets: {targets.shape}')
        loss = nn.CrossEntropyLoss(reduction='none')(logits, targets).detach().cpu()  # cifar10: logits - torch.Size([256, 10]), targets - torch.Size([256])

        data_importance['entropy'][idx] = entropy
        data_importance['loss'][idx] = loss
    
    # print(f'data_importance: {data_importance}')

"""Calculate td metrics"""
def training_dynamics_metrics(td_log, dataset, data_importance):
    targets = []
    data_size = len(dataset)

    for i in range(data_size):
        _, (_, y) = dataset[i]
        targets.append(y)
    targets = torch.tensor(targets)
    data_importance['targets'] = targets.type(torch.int32)

    data_importance['correctness'] = torch.zeros(data_size).type(torch.int32)
    data_importance['forgetting'] = torch.zeros(data_size).type(torch.int32)
    data_importance['last_correctness'] = torch.zeros(data_size).type(torch.int32)
    data_importance['accumulated_margin'] = torch.zeros(data_size).type(torch.float32)

    def record_training_dynamics(td_log):
        output = torch.exp(td_log['output'].type(torch.float))  # cifar10: torch.Size([256, 10]), organamnist: torch.Size([256, 11])
        predicted = output.argmax(dim=1)                        # cifar10: torch.Size([256]), organamnist: torch.Size([256])
        index = td_log['idx'].type(torch.long)                  # cifar10: torch.Size([256]), organamnist: torch.Size([256])

        label = targets[index]                              # cifar10: torch.Size([256]), organamnist: torch.Size([256, 1])
        label = label.squeeze()                             # organamnist: torch.Size([256]) (med)

        correctness = (predicted == label).type(torch.int)  # cifar10: torch.Size([256])
        data_importance['forgetting'][index] += torch.logical_and(data_importance['last_correctness'][index] == 1, correctness == 0)  # cifar10: torch.Size([256])
        data_importance['last_correctness'][index] = correctness
        data_importance['correctness'][index] += data_importance['last_correctness'][index]

        batch_idx = range(output.shape[0])
        target_prob = output[batch_idx, label]
        output[batch_idx, label] = 0
        other_highest_prob = torch.max(output, dim=1)[0]
        margin = target_prob - other_highest_prob
        data_importance['accumulated_margin'][index] += margin

    for i, item in enumerate(td_log):
        if i % 10000 == 0:
            print('training_dynamics_metrics, td_log', i)

        record_training_dynamics(item)

"""Calculate td metrics"""
def EL2N(td_log, dataset, data_importance, early_max_epoch=10, later_min_epoch=100, later_max_epoch=120):
    targets = []
    data_size = len(dataset)
    num_later_epochs = later_max_epoch - later_min_epoch

    for i in range(data_size):
        _, (_, y) = dataset[i]
        targets.append(y)
    targets = torch.tensor(targets)  # cifar10: torch.Size([50000]), organamnist: torch.Size([34561, 1])
    targets = targets.squeeze()      # med; organamnist: torch.Size([34561])
    # breakpoint()
    data_importance['targets'] = targets.type(torch.int32)
    data_importance['el2n'] = torch.zeros(data_size).type(torch.float32)

    # 分别存储 early epoch 和 later epoch 的 el2n_score
    data_importance['el2n_early'] = torch.zeros((early_max_epoch, data_size)).type(torch.float32)   # torch.Size([10, 13932])
    data_importance['el2n_later'] = torch.zeros((num_later_epochs, data_size)).type(torch.float32)  # torch.Size([10, 13932])
    # breakpoint()

    l2_loss = torch.nn.MSELoss(reduction='none')

    def record_training_dynamics(item, epoch_idx, phase):  # for each item in td_log
        output = torch.exp(item['output'].type(torch.float))
        predicted = output.argmax(dim=1)
        index = item['idx'].type(torch.long)

        label = targets[index]
        label_onehot = torch.nn.functional.one_hot(label, num_classes=num_classes)

        el2n_score = torch.sqrt(l2_loss(label_onehot, output).sum(dim=1))  # ||.||=sqrt(sum(.)2)

        # 将当前 epoch 的 el2n_score 保存到对应的位置
        if phase == 'early':
            data_importance['el2n_early'][epoch_idx, index] = el2n_score  # data_importance['el2n_early'].shape = [10, 13932]
        else:
            data_importance['el2n_later'][epoch_idx-later_min_epoch, index] = el2n_score  # data_importance['el2n_later'].shape = [10, 13932]

    
    early_epoches = []
    later_epoches = []
    epoch_list = []
    print(f'td_log length: {len(td_log)}')  # 10800: organsmnist (all-data) Total iterations
    # td_log_epoch = list(item['epoch'] for _, item in enumerate(td_log)) # 54个0, 54个1, ..., 54个199
    for i, item in enumerate(td_log):
        # item: {'epoch': 0, 'iteration': 0, 'idx': tensor([...]), 'output': tensor([[..],..., [..]])}
        # item['idx'].shape = torch.Size([256]); item['output'].shape = torch.Size([256, 11]) (num_classes)
        epoch_idx = item['epoch']  # 获取当前 epoch 索引
        
        # el2n_score_list = []
        if i % 10000 == 0:
            print('EL2N, td_log', i)
        
        # 记录当前 epoch 的训练动态
        if epoch_idx < early_max_epoch:  # 0~9
            early_epoches.append(epoch_idx)
            record_training_dynamics(item, epoch_idx, 'early')
        elif (epoch_idx >= later_min_epoch) and (epoch_idx < later_max_epoch):  # 100~109
            later_epoches.append(epoch_idx)
            record_training_dynamics(item, epoch_idx, 'later')
        if epoch_idx == later_max_epoch:
            print('early_epoches:', len(early_epoches))     # 540 (54*10)=(Iterations_per_epoch * early_max_epoch)
            print('later_epoches:', len(later_epoches))     # 540 (54*10)=(Iterations_per_epoch * later_epochs)

        # # >>>>>>>>>>>>>>> 1. calculate Variance
        # # 如果达到了early_max_epoch，计算方差
        # if epoch_idx == early_max_epoch:
        #     print('early_epoches:', len(early_epoches))     # 540 (54*10)=(Iterations_per_epoch * early_max_epoch)
        #     # 计算每个样本在 early_max_epoch 个 epoch 的 el2n_score 的方差
        #     el2n_var = torch.var(data_importance['el2n_early'], dim=0)
        #     data_importance['el2n'] = el2n_var  # torch.Size([13932]) = len(trainset)

        #     return 
        # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # # >>>>>>>>>>>>>>> 2. early epoch and later epoch
        if epoch_idx == early_max_epoch:    # 计算每个样本在 early_max_epoch 个 epoch 的 el2n_score 的方差
            el2n_var_e = torch.var(data_importance['el2n_early'], dim=0)  # torch.Size([13932]) = len(trainset)
            el2n_var_e_norm = F.normalize(el2n_var_e.unsqueeze(0), p=1, dim=1)  # 在第 0 维上增加一个维度后再归一化 
            # breakpoint()
        elif epoch_idx == later_max_epoch:  # 计算每个样本在 num_later_epoch 个 epoch 的 el2n_score 的方差
            el2n_var_l = torch.var(data_importance['el2n_later'], dim=0)
            el2n_var_l_norm = F.normalize(el2n_var_l.unsqueeze(0), p=1, dim=1)  # torch.Size([1, 13932])
            el2n_var = torch.squeeze(el2n_var_e_norm + el2n_var_l_norm)     # torch.Size([13932])
            data_importance['el2n'] = el2n_var
            # # mean
            # el2n_var = torch.stack((el2n_var_e, el2n_var_l), dim=0)       # torch.Size([2, 13932])
            # data_importance['el2n'] = el2n_var.mean(dim=0)                # torch.Size([13932])
            
            return
        # # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

        # # 记录当前 epoch 的训练动态
        # record_training_dynamics(item, epoch_idx)
        
#########################

GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_identical = transforms.Compose([transforms.ToTensor(),])

data_dir =  os.path.join(args.data_dir, dataset)
print(f'dataset: {dataset}')
if dataset == 'cifar10':
    trainset = CIFARDataset.get_cifar10_train(data_dir, transform = transform_identical)
elif dataset == 'cifar100':
    trainset = CIFARDataset.get_cifar100_train(data_dir, transform = transform_identical)
elif dataset == 'svhn':
    trainset = SVHNDataset.get_svhn_train(data_dir, transform = transform_identical)
elif args.dataset == 'cinic10':
    trainset = CINIC10Dataset.get_cinic10_train(data_dir, transform = transform_identical)
else:  # MedMNIST
    DataClass = getattr(medmnist, info['python_class'])
    data_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])])
    trainset = DataClass(split='train', transform=data_transform, download=download, as_rgb=as_rgb)

trainset = IndexDataset(trainset)
print(len(trainset))

data_importance = {}

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=args.batch_size, shuffle=False, num_workers=16, drop_last=True)  # add drop_last

model = resnet('resnet18', num_classes=num_classes, device=device)
model = model.to(device)

print(f'Ckpt path: {ckpt_path}.')
checkpoint = torch.load(ckpt_path)['model_state_dict']
model.load_state_dict(checkpoint)
model.eval()

with open(td_path, 'rb') as f:
     pickled_data = pickle.load(f)

training_dynamics = pickled_data['training_dynamics']  # td_log
early_max_epoch = 10
later_min_epoch = 100
later_max_epoch = 110

post_training_metrics(model, trainloader, data_importance, device)
training_dynamics_metrics(training_dynamics, trainset, data_importance)
EL2N(training_dynamics, trainset, data_importance, early_max_epoch=early_max_epoch, later_min_epoch=later_min_epoch, later_max_epoch=later_max_epoch)

print(f'Saving data score at {data_score_path}')
with open(data_score_path, 'wb') as handle:
    pickle.dump(data_importance, handle)