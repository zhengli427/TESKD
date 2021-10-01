import argparse
import os
import random
import shutil
import time

import torch.distributed as dist
import torch.nn as nn

import distill_loss
import models
import models_imagenet
from dataset import create_loader
from utils import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.')
    parser.add_argument('--final_dir', type=str, default='.')
    parser.add_argument('--data', default='CIFAR100', type=str)
    parser.add_argument('--random_seed', default=27, type=int)
    parser.add_argument('--name', default='', type=str)
    parser.add_argument('--model_name', default='resnet_our', type=str)
    parser.add_argument('--network_name', default='cifarresnet18', type=str)

    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--scheduler', default='step', type=str, help='step|cos')
    parser.add_argument('--schedule', default=[100, 150], type=int, nargs='+')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--lr_decay', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--save_model', default=True, type=bool)

    parser.add_argument('--fea_weight', default=1e-7, type=float)
    parser.add_argument('--kd_weight', default=0.8, type=float)
    parser.add_argument('--ce_weight', default=0.2, type=float)

    parser.add_argument('--backbone_weight', default=1.0, type=float)
    parser.add_argument('--b1_weight', default=1.0, type=float)
    parser.add_argument('--b2_weight', default=1.0, type=float)
    parser.add_argument('--b3_weight', default=1.0, type=float)

    # parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')

    args = parser.parse_args()
    args.nprocs = torch.cuda.device_count()

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    args_path = args.final_dir
    path_log = os.path.join(args_path, 'logs', f'{args.network_name}', f'{args.name}')
    if not os.path.exists(path_log):
        os.makedirs(path_log)
    logger = create_logging(os.path.join(path_log, 'logs.txt'))

    this_dir = os.path.dirname(__file__)
    if args.data == 'CIFAR100':
        mdl_path = 'models'
    else:
        mdl_path = 'models_imagenet'
    shutil.copy2(os.path.join(this_dir, mdl_path, args.model_name + '.py'), path_log)
    args.batch_size = int(args.batch_size / args.nprocs)

    train_loader, test_loader, args.num_classes = create_loader(args.batch_size, args.data_dir, args.data)

    model = eval(mdl_path + '.' + args.model_name + "." + args.network_name)(num_classes=args.num_classes)
    model = model.cuda()
    # model = nn.DataParallel(model).cuda()
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    criterion_ce = nn.CrossEntropyLoss().cuda()
    # criterion_ce = nn.CrossEntropyLoss().cuda(args.local_rank)

    criterion_kd = distill_loss.KD.KL_Loss(temperature=3).cuda()
    # criterion_kd = distill_loss.KD.KL_Loss(temperature=3).cuda(args.local_rank)

    criterion_fea = distill_loss.Fea_Loss()

    criterion = [criterion_ce, criterion_kd, criterion_fea]
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          nesterov=True,
                          weight_decay=args.weight_decay)
    scheduler = lr_scheduler(optimizer, args.scheduler, args.schedule, args.lr_decay, args.epoch)

    sota_acc = 0
    for epoch in range(1, args.epoch + 1):
        # s = time.time()
        loss, train_acc1, train_acc_b1, train_acc_b2, train_acc_b3 \
            = train(model, optimizer, criterion, train_loader, args)

        scheduler.step()
        test_acc1, test_b1, test_b2, test_b3 = test(model, test_loader)
        logger.info(
            'Epoch: {0:>2d}|Train Loss: {1:2.4f}| Test Acc: {2:.4f}| Test_b1 Acc: {3:.4f}| Test_b2 Acc: {4:.4f}| '
            'Test_b3 Acc: {5:.4f}'.format(epoch, loss, test_acc1, test_b1, test_b2, test_b3))
        if test_acc1 > sota_acc and epoch > 150:
            sota_acc = test_acc1
            logger.info('----SOTA Accuarcy----')

        if args.save_model:
            if test_acc1 > sota_acc:
                if epoch > 150:
                    final_model_state_file = os.path.join(path_log, 'final_state.pth')
                    torch.save(model.state_dict(), final_model_state_file)
                    sota_acc = test_acc1
                    logger.info('----SOTA Accuarcy----')


if __name__ == '__main__':
    main()
