import logging
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler as GradScaler
from torch.cuda.amp import autocast as autocast
from tqdm import tqdm

scaler = GradScaler()

def create_logging(path_log):
    logger = logging.getLogger('Result_log')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(path_log)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)
    return logger


def train(model, optimizer, criterion, train_loader, args):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top1_b1 = AverageMeter()
    top1_b2 = AverageMeter()
    top1_b3 = AverageMeter()

    criterion_ce, criterion_kd, criterion_fea = criterion
    with tqdm(total=len(train_loader)) as t:
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, targets = inputs.cuda(), targets.cuda()

            with autocast():
                out, b1_out, b2_out, b3_out, final_fea, b3_fea, b2_fea, b1_fea = model(inputs)

                loss_model = args.backbone_weight * criterion_ce(out, targets)
                loss_model += args.b1_weight * criterion_ce(b1_out, targets)
                loss_model += args.b2_weight * criterion_ce(b2_out, targets)
                loss_model += args.b3_weight * criterion_ce(b3_out, targets)

                loss_kd = criterion_kd(b1_out, out)
                loss_kd += criterion_kd(b2_out, out)
                loss_kd += criterion_kd(b3_out, out)

                loss_fea = criterion_fea(b1_fea, final_fea.detach())
                loss_fea += criterion_fea(b2_fea, final_fea.detach())
                loss_fea += criterion_fea(b3_fea, final_fea.detach())

                loss = args.ce_weight * loss_model + args.kd_weight * loss_kd + args.fea_weight * loss_fea
                # print(loss)
                t.update()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        #loss.backward()
        #optimizer.step()

        batch_size = targets.size(0)
        losses.update(loss.item(), batch_size)
        acc1, _ = accuracy(out, targets, topk=(1, 5))
        top1.update(acc1, batch_size)

        acc1_b1, _ = accuracy(b1_out, targets, topk=(1, 5))
        top1_b1.update(acc1_b1, batch_size)

        acc1_b2, _ = accuracy(b2_out, targets, topk=(1, 5))
        top1_b2.update(acc1_b2, batch_size)

        acc1_b3, _ = accuracy(b3_out, targets, topk=(1, 5))
        top1_b3.update(acc1_b3, batch_size)

    return losses.avg, top1.avg, top1_b1.avg, top1_b2.avg, top1_b3.avg


def test(model, test_loader):
    model.eval()
    top1 = AverageMeter()
    top1_b1 = AverageMeter()
    top1_b2 = AverageMeter()
    top1_b3 = AverageMeter()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            batch_size = targets.size(0)
            inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                out, out_b1, out_b2, out_b3, _, _, _, _ = model(inputs)

            acc1, _ = accuracy(out, targets, topk=(1, 5))
            top1.update(acc1, batch_size)

            acc1_b1, _ = accuracy(out_b1, targets, topk=(1, 5))
            top1_b1.update(acc1_b1, batch_size)

            acc1_b2, _ = accuracy(out_b2, targets, topk=(1, 5))
            top1_b2.update(acc1_b2, batch_size)

            acc1_b3, _ = accuracy(out_b3, targets, topk=(1, 5))
            top1_b3.update(acc1_b3, batch_size)

    return top1.avg, top1_b1.avg, top1_b2.avg, top1_b3.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch in args.schedule:
        args.lr = args.lr * args.lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

def lr_scheduler(optimizer, scheduler, schedule, lr_decay, total_epoch):
    optimizer.zero_grad()
    optimizer.step()
    if scheduler == 'step':
        return optim.lr_scheduler.MultiStepLR(optimizer, schedule, gamma=lr_decay)
    elif scheduler == 'cos':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, total_epoch)
    else:
        raise NotImplementedError('{} learning rate is not implemented.')
