from datetime import datetime
import os
import os.path as osp
import math
import random
import timeit
import numpy as np
import torch
import torch.nn.functional as F
import pytz
from tensorboardX import SummaryWriter
import torch.nn as nn
from torch.autograd import Variable
import tqdm
import socket
from skimage.exposure import match_histograms

bceloss = torch.nn.BCELoss()
mseloss = torch.nn.MSELoss()
softmax = torch.nn.Softmax(-1)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Multi_DiceLoss(nn.Module):
    def __init__(self, class_num=7, smooth=0.001):
        super(Multi_DiceLoss, self).__init__()
        self.smooth = smooth
        self.class_num = class_num

    def forward(self, input, target):
        input = torch.softmax(input, dim=1)
        Dice = Variable(torch.Tensor([0]).float()).cuda()
        have_class = 0
        for i in range(1, self.class_num):
            if torch.sum(target == i) == 0:
                continue
            have_class += 1
            input_i = input[:, i, :, :]
            target_i = (target == i).float()
            # target_i = target[:, i, :, :]
            intersect = (input_i * target_i).sum()
            union = torch.sum(input_i) + torch.sum(target_i)
            dice = (2 * intersect + self.smooth) / (union + self.smooth)
            Dice += dice
        dice_loss = 1 - Dice / have_class
        return dice_loss

def cosine_similarity_1d(tensor1, tensor2):
    tensor1 = tensor1.float()
    tensor2 = tensor2.float()
    dot_product = torch.dot(tensor1, tensor2)
    norm1 = torch.norm(tensor1)
    norm2 = torch.norm(tensor2)
    cosine_sim = dot_product / (norm1 * norm2)
    return cosine_sim

class Trainer(object):

    def __init__(self, cuda, model, lr, val_loader, train_loader, out, max_epoch, optim, stop_epoch=None,
                 lr_decrease_rate=0.1, interval_validate=None, batch_size=8):
        self.cuda = cuda
        self.model = model
        self.optim = optim
        self.lr = lr
        self.lr_decrease_rate = lr_decrease_rate
        self.batch_size = batch_size

        self.val_loader = val_loader
        self.train_loader = train_loader
        self.time_zone = 'Asia/Hong_Kong'
        self.timestamp_start = \
            datetime.now(pytz.timezone(self.time_zone))

        if interval_validate is None:
            self.interval_validate = int(10)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss_seg',
            'train/cup_dice',
            'train/disc_dice',
            'valid/loss_CE',
            'valid/cup_dice',
            'valid/disc_dice',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        log_dir = os.path.join(self.out, 'tensorboard',
                               datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        self.writer = SummaryWriter(log_dir=log_dir)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.stop_epoch = stop_epoch if stop_epoch is not None else max_epoch
        self.best_disc_dice = 0.0
        self.running_loss_tr = 0.0
        self.running_adv_diff_loss = 0.0
        self.running_adv_same_loss = 0.0
        self.best_mean_dice = 0.0
        self.best_epoch = -1

    def validate(self):
        training = self.model.training
        self.model.eval()

        val_loss = 0
        with torch.no_grad():

            for batch_idx, sample in tqdm.tqdm(
                    enumerate(self.val_loader), total=len(self.val_loader),
                    desc='Valid iteration=%d' % self.iteration, ncols=80,
                    leave=False):

                image = sample['image']
                label = sample['label']
                domain_code = sample['dc']

                data = image.cuda()
                target_map = label.cuda()
                domain_code = domain_code.cuda()

                with torch.no_grad():
                    predictions, v, mixm, md, HDomain = self.model(data,domain_code,domain_code,'test')
                Diceloss = Multi_DiceLoss()
                # celoss = nn.CrossEntropyLoss()
                # print(predictions.size(), target_map.size())
                loss_seg = Diceloss(predictions,target_map)
                # loss_cls = mseloss(softmax(domain_predict), domain_code)
                loss_data = 1-loss_seg.data.item()
                # print(loss_data)
                # if np.isnan(loss_data):
                #     raise ValueError('loss is nan while validating')
                val_loss += loss_data

            val_loss /= len(self.val_loader)
            self.writer.add_scalar('val_data/loss', val_loss, self.epoch * (len(self.train_loader)))
            # self.writer.add_scalar('val_data/loss_cls', loss_cls.data.item(), self.epoch * (len(self.train_loader)))
            print("Dice：{}".format(val_loss))

            mean_dice = val_loss
            # is_best = mean_dice > self.best_mean_dice
            # if is_best:
            self.best_epoch = self.epoch + 1
            self.best_mean_dice = mean_dice

            torch.save({
                # 'epoch': self.epoch,
                'iteration': self.iteration,
                'arch': self.model.__class__.__name__,
                'optim_state_dict': self.optim.state_dict(),
                'model_state_dict': self.model.state_dict(),
                'learning_rate_gen': get_lr(self.optim),
                'best_mean_dice': self.best_mean_dice,
            }, osp.join(self.out, f'checkpoint_{self.epoch}_{100 * mean_dice:.4f}.pth.tar'))
            

            if training:
                self.model.train()

    def train_epoch(self):
        self.model.train()
        self.running_seg_loss = 0.0
        self.running_total_loss = 0.0
        self.running_cup_dice_tr = 0.0
        self.running_disc_dice_tr = 0.0
        self.running_cls_loss = 0

        start_time = timeit.default_timer()
        for batch_idx, sample in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            self.iteration = iteration

            assert self.model.training
            self.optim.zero_grad()

            image = None
            label = None
            domain_code = None
            for domain in sample:
                if image is None:
                    image = domain['image']
                    label = domain['label']
                    domain_code = domain['dc']
                else:
                    image = torch.cat([image, domain['image']], 0)
                    label = torch.cat([label, domain['label']], 0)
                    domain_code = torch.cat([domain_code, domain['dc']], 0)

            imagenp = image.numpy()
            imagemix = np.flip(imagenp, axis=0)
            domainnp = domain_code.numpy()

            for i in range(0,2):
                imagemix[i, :, :, :] = match_histograms(imagemix[i, :, :, :], domainnp[i, :, :, :])

            imagemix = torch.from_numpy(imagemix.copy())

            image = image.cuda()
            target_map = label.cuda()
            domain_code = domain_code.cuda()
            imagemix = imagemix.cuda()

            target_map = torch.cat([target_map, target_map], dim=0)

            # print(image.size(), target_map.size())
            output, v, mixm, md, HDomain,mixv = self.model(image, domain_code, imagemix, 'train')

            v = torch.var(v, dim=(2, 3))
            f = torch.randn_like(v)
            a = random.uniform(0, 1)
            v = a*v+(1-a)*f
            var_loss = math.log((math.exp(cosine_similarity_1d(v[0], v[1]) / 0.1) + math.exp(cosine_similarity_1d(v[2], v[1]) / 0.1) + math.exp(cosine_similarity_1d(v[0], v[2]) / 0.1)))
            mixv = torch.var(mixv, dim=(2, 3))
            v = torch.cat([v, mixv], dim=0)

            matrix1 = torch.matmul(v, v.mT)/(torch.norm(v,p = 2)*torch.norm(v.mT,p = 2))
            mi1 = torch.eye(matrix1.shape[0], device=matrix1.device)
            matrix2 = torch.matmul(v.mT, v)/(torch.norm(v.mT,p = 2)*torch.norm(v,p = 2))
            mi2 = torch.eye(matrix2.shape[0], device=matrix2.device)
            var_loss += torch.norm(matrix1-mi1,p = 2)+torch.norm(matrix2-mi2,p = 2)

            rm = math.exp(cosine_similarity_1d(md[0], md[1]) / 0.1) + math.exp(cosine_similarity_1d(md[2], md[1]) / 0.1) + math.exp(cosine_similarity_1d(md[0], md[2]) / 0.1)
            mean_loss = rm / (rm + math.exp(cosine_similarity_1d(mixm[0], md[0]) / 0.1) + math.exp(cosine_similarity_1d(mixm[1], md[1]) / 0.1) + math.exp(cosine_similarity_1d(mixm[2], md[2]) / 0.1))

            criterion = nn.CrossEntropyLoss()

            loss_seg = criterion(output, target_map)
            # print(output.size(), target_map.size())

            l1_distance = torch.nn.L1Loss()
            drc_seg = l1_distance(HDomain,domain_code)

            self.running_seg_loss += loss_seg.item()
            loss_data = loss_seg.data.item()

            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')

            loss = loss_seg+(var_loss-math.log(mean_loss)+drc_seg)*0.1
            
            loss.backward()
            self.optim.step()
            # self.optim.zero_grad()

            self.writer.add_scalar('train_gen/loss', loss_data, iteration)
            self.writer.add_scalar('train_gen/loss_seg', loss_seg.data.item(), iteration)

        self.running_seg_loss /= len(self.train_loader)
        # self.running_cls_loss /= len(self.train_loader)
        stop_time = timeit.default_timer()

        print('\n[Epoch: %d] lr:%f,  Average segLoss: %f, Average clsLoss: %f, Execution time: %.5f' %
              (self.epoch, get_lr(self.optim), self.running_seg_loss, self.running_cls_loss, stop_time - start_time))

    def train(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Train', ncols=80):
            torch.cuda.empty_cache()
            self.epoch = epoch
            self.train_epoch()
            if self.stop_epoch == self.epoch:
                print('Stop epoch at %d' % self.stop_epoch)
                break

            if (epoch + 1) % (self.max_epoch//2) == 0:
                _lr_gen = self.lr * self.lr_decrease_rate
                for param_group in self.optim.param_groups:
                    param_group['lr'] = _lr_gen
            self.writer.add_scalar('lr', get_lr(self.optim), self.epoch * (len(self.train_loader)))
            if (self.epoch + 1) % self.interval_validate == 0 or self.epoch == 0:
                self.validate()
        self.writer.close()



