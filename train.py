# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from os import path as osp
from tensorboardX import SummaryWriter
import torch
from torch import nn
import torch.nn.functional as F
from math import log10
from vgg import Vgg16
from model import pix2vec
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
import torch.distributed as dist
from torchvision.utils import make_grid
from argparse import Namespace
from collections import Counter

def get_args():
    args = Namespace(
        vis_freq=250, logdir='', eval_freq=1000, 
        ngpus=4, bSz=8, ckptPath='',
        milestones = [800000, 900000], gamma = 0.1, # scheduler
        lr=2e-4, wd=0., # optimizer
        max_iter=1000000,

        # model params
        nParam=29, nMasks=10, iSz=128, mlpHSz=128,
        noGN=False, GN_before_sigmoid=False, use_relu=False, bnk=512,
        normalization='GN', canvasInit='black',

        # loss
        recW=0., contentW=1., styleW=25000.0, 
        content_layers=[0,1,2,3], style_layers=[0,1,2,3], 

    )
    return args

def train(args):
    trainer = Trainer(args)
    trainer.train()
    
class Trainer():
    def __init__(self, args):
        self.setup_environ(args)
        self.log = True
        self.distributed = False
        self.global_rank = 0
        if args.ngpus > 1:
            self.setup_distributed(args)
        
#         dataset = ImageFolder('/datasets01/CelebA/CelebA/072017/')
        dataset = torch.load('celeba_dset.pth')
        nimages = len(dataset)
        nval = 1000
        inds = list(range(nimages))
        train_inds = inds[1000:]
        val_inds = inds[:1000]
        
        # TODO: add jitter augmentation ?
        self.train_transform = T.Compose([
            T.Resize(args.iSz),
            T.RandomHorizontalFlip(0.5),
            T.CenterCrop(args.iSz),
            T.ToTensor()
        ])
        self.val_transform = T.Compose([
            T.Resize(args.iSz),
            T.CenterCrop(args.iSz),
            T.ToTensor()
        ])
        
        self.train_dset = Subset(dataset, train_inds, transform=self.train_transform)
        self.val_dset = Subset(dataset, val_inds, transform=self.val_transform)
        
        if self.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dset)
        else:
            train_sampler = None

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dset, batch_size=args.bSz, shuffle=(train_sampler is None),
            num_workers=10, pin_memory=True, sampler=train_sampler)
        
        self.val_loader = torch.utils.data.DataLoader(self.val_dset, batch_size=50, num_workers=10) # 1000 samples
        
        self.model = pix2vec(args)
        # change normalization
        if args.normalization != 'None':
            kwargs = {}
            if args.normalization == 'GN':
                kwargs['nGroups'] = 32
            change_norm(self.model, normType=args.normalization, verbose=0, **kwargs)
        
        self.model = self.model.cuda()
        
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.local_rank], output_device=self.local_rank,
                find_unused_parameters=True
            )
        if self.log:
            print(f'| {self.model}')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.999), amsgrad=False)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.milestones, gamma=args.gamma)
        
        self.useVgg = args.contentW > 0 or args.styleW > 0
        if self.useVgg > 0:
            self.vgg = Vgg16(requires_grad=False).to('cuda').eval() # TODO: eval ???
            self.mse_loss = torch.nn.MSELoss()
            
        self.iteration = 0
        self.bestPSNR = 0
        
        if self.log:
            self.writer = SummaryWriter(args.logdir)
            self.writer.add_text('args', str(args), 0)
        
        self.args = args
        
        if args.ckptPath:
            self.load(args.ckptPath)
            
        
        print(' | Done init trainer')
        
    def setup_environ(self, args):
        self.job = False
        try:
            import submitit
            self.job_env = submitit.JobEnvironment()
            args.logdir = args.logdir.replace('%j', str(self.job_env.job_id))
            self.job = True
        except:
            self.job_env = None
            pass
        
    def setup_distributed(self, args):
        if args.ngpus > 1:
            assert self.job_env is not None
            self.jobid = self.job_env.job_id
            self.local_rank = self.job_env.local_rank
            self.global_rank = self.job_env.global_rank
            self.num_tasks = self.job_env.num_tasks
            
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(
                backend='nccl',
                init_method=args.dist_url,
                world_size=self.num_tasks,
                rank=self.global_rank,
            )
            print('| Successfully setup distributed')
            self.distributed = True
        
        self.log = self.global_rank == 0
        
    def train(self):
        args = self.args
        
        while self.iteration < args.max_iter:
            for i, (ims, _) in enumerate(self.train_loader):
                if self.iteration > args.max_iter:
                    break
                output = {}
                output['ims'] = ims
                
                losses, output = self.model_step(output)
                
                if self.iteration % 10 == 0 and self.log:
                    for k,v in losses.items():
                        self.writer.add_scalar(f'G_losses/{k}', v.cpu().detach(), self.iteration)
                    self.writer.add_scalar('Metrics/lr', self.scheduler.get_last_lr()[0], self.iteration)
                    
                if self.iteration % args.vis_freq == 0 and self.log:
                    self.writer.add_images('Train/input', ims[:4].cpu(), self.iteration)
                    self.writer.add_images('Train/output', output['recs'][:4].cpu(), self.iteration)
                    
                if self.iteration % args.eval_freq == 0:
                    self.save()
                    L1, L2, PSNR = self.validate()
                    if PSNR > self.bestPSNR:
                        self.bestPSNR = PSNR
                        self.save(name='best')
                    
                self.iteration += 1
            
    def model_step(self, output):
        args = self.args
        losses = {}
        self.model.train()
        self.optimizer.zero_grad()
        ims = output['ims'].cuda(non_blocking=True)
        recs = self.model(ims)
        output['recs'] = recs.detach()
        
        if args.recW > 0.:
            losses['G_rec'] = args.recW * F.l1_loss(recs, ims)
        
        if self.useVgg:
            features_y = self.vgg(normalize_batch(recs))
            features_x = self.vgg(normalize_batch(ims))
            
            if args.contentW > 0. and len(args.content_layers):
                content_loss = 0
                for i, (ft_x, ft_y) in enumerate(zip(features_x, features_y)):
                    if i in args.content_layers:
                        content_loss += self.mse_loss(ft_x, ft_y)
                losses['G_content'] = args.contentW * content_loss
                
            if args.styleW > 0. and len(args.style_layers):
                style_loss = 0
                for i,(ft_x, ft_y) in enumerate(zip(features_x, features_y)):
                    if i in args.style_layers:
                        style_loss += self.mse_loss(gram_matrix(ft_x), gram_matrix(ft_y))
                losses['G_style'] = args.styleW * style_loss
            
        losses = self.reduce(losses)
        
        loss = sum(losses.values()).mean()
        loss.backward()
        
        self.optimizer.step()
        self.scheduler.step()
        self.model.eval()
        
        return losses, output
        
    def load(self, ckptPath):
        args = self.args
        x = torch.load(ckptPath)

        self.model.load_state_dict(x['model'])
        self.optimizer.load_state_dict(x['optimizer'])
        self.scheduler.load_state_dict(x['scheduler'])
        if len(args.milestones):
            self.scheduler.milestones = Counter(args.milestones)
        self.iteration = x['iteration']
        self.bestPSNR = x['bestPSNR']
        print(f'Loaded ckpt from {ckptPath} with {self.iteration} iterations and {self.bestPSNR} PSNR')
        
    def save(self, name=''):
        if not self.log:
            return
        args = self.args
        d = {
            'model':self.model.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'iteration':self.iteration,
            'bestPSNR':self.bestPSNR,
            'args':self.args,
        }
        if name:
            name = '_' + name
        
        torch.save(d, osp.join(args.logdir, f'ckpt{name}.pth'))
        
    def validate(self):
        self.model.eval()
        L1 = []; L2 = []
        for i, (ims, _) in enumerate(self.val_loader):
            ims = ims.cuda(non_blocking=True)
            with torch.no_grad():
                recs = self.model(ims)
                l1 = F.l1_loss(recs, ims, reduction='none').view(ims.size(0), -1).mean(1)
                l2 = F.mse_loss(recs, ims, reduction='none').view(ims.size(0), -1).mean(1)
                L1.append(l1); L2.append(l2)
        L1 = torch.cat(L1).mean()
        L2 = torch.cat(L2).mean()
        PSNR = 10 * log10(1 / L2)
            
        if self.log:
            self.writer.add_scalar('Val/L1', L1.item(), self.iteration)
            self.writer.add_scalar('Val/L2', L2.item(), self.iteration)
            self.writer.add_scalar('Val/PSNR', PSNR, self.iteration)
            
        # visualize masks
        N = 4
        with torch.no_grad():
            output = self.model(ims[:N], return_=True)
        
        if self.log:
            masks = torch.stack(output['allMasks']).transpose(0,1).contiguous()
            for i in range(N):
                x = make_grid(masks[i], nrow=10)
                self.writer.add_image(f'Val/masks_{i}', x.cpu(), self.iteration)
        
        return L1, L2, PSNR
        
    def reduce(self, x):
        if not self.distributed:
            return x
        if isinstance(x, float) or isinstance(x, int):
            return x
        elif isinstance(x, dict):
            for k,v in x.items():
                x[k] = self.reduce(v)
            return x
        rt = x.clone()
        dist.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= self.num_tasks
        return rt

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return (batch - mean) / std

class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        im, labels = self.dataset[self.indices[idx]]
        if self.transform is not None:
            im = self.transform(im)
        return im, labels

    def __len__(self):
        return len(self.indices)
    
    
def change_norm(net, normType='None', verbose=0, **kwargs):
    """ 
    Replace batchnorm modules with groupNorm ones 
    
    """
    def applyNorm(net):
        for name in net._modules:
            m = getattr(net, name)
            className = m.__class__.__name__
            if 'BatchNorm' in className:
                if normType == 'LN':
                    nFeatures = m.num_features
                    if verbose:
                        print('Changing batchnorm {} to layerNorm in {}'.format(name,net.__class__.__name__))
                    setattr(net, name, nn.GroupNorm(num_groups=1, num_channels=nFeatures))
                elif normType == 'GN':
                    nFeatures = m.num_features
                    nGroups = min(kwargs['nGroups'],nFeatures)
                    if verbose:
                        print('Changing batchnorm {} to groupNorm in {}'.format(name, net.__class__.__name__))
                    setattr(net, name, nn.GroupNorm(num_groups=nGroups, num_channels=nFeatures))
                elif normType == 'None':
                    if verbose:
                        print('Changing batchnorm {} to sequential in {}'.format(name,net.__class__.__name__))
                    setattr(net, name, nn.Sequential())
                else:
                    raise ValueError('Unknown normalization {}'.format(normType))
    net.apply(applyNorm)