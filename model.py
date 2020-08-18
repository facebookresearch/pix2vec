import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

class pix2vec(nn.Module):
    def __init__(self, args):
        super(pix2vec, self).__init__()
        self.args = args
        
        self.backbone = resnetEncoder(in_c=6, bnk=args.bnk)
        
        self.maskParam_fc = get_maskParam_fc(args)# module to regress mask parameters
        self.color_fc = get_color_fc(args) # module to regress color
        
        additional_dims = 2
        in_channels = args.nParam + additional_dims
        out_channels = 1
        hidden_size = args.mlpHSz
        use_GN = not args.noGN
        use_GN_before_sigmoid = args.GN_before_sigmoid
        use_relu = args.use_relu
        
        self.maskgen = maskgen(in_channels, out_channels, 
                               hidden_size=hidden_size, 
                               use_GN=use_GN, 
                               use_GN_before_sigmoid=use_GN_before_sigmoid, 
                               use_relu=use_relu, 
                               use_sinus_tanh=False)

        self.args = args
        
    def get_grid(self, size, bSz, device):
        if hasattr(self, 'grid'):
            grid = self.grid
            if grid.size(0) == bSz and grid.size(-1) == size:
                return self.grid.to(device)
        return get_grid_coordinates(size, bSz, device)

    def forward(self, targets=None, return_=False, nMasks=None, **kwargs):
        args = self.args
        bSz, nc, iSz, _ = targets.size()
        device = targets.device

        nParam = args.nParam
        nMasks = args.nMasks if nMasks is None else nMasks
        canvas = initCanvas(iSz, bSz, device, mode=args.canvasInit, nc=3)
        
        if return_:
            allCanvas=[];  allMaskParams=[]; allColors=[]; allMasks=[];

        for i in range(nMasks):
            inp = torch.cat([canvas.detach(), targets], 1)
            
            x = self.backbone(inp)
            
            maskParams = self.maskParam_fc(x).view(bSz,nParam,1,1).expand(bSz, nParam, iSz, iSz)
            
            inp = [maskParams]
            
            # get xy grid
            grid = self.get_grid(iSz, bSz, device)
            inp.append(grid)
            
            mask = self.maskgen(torch.cat(inp, 1))# generate noise using mask parameters + xy-grid and optionally
            colors = self.color_fc(x)
            colorMap = colors.view(-1, nc, 1, 1).expand(-1,-1, iSz, iSz)

            canvas = canvas*(1-mask) + mask*colorMap
            
            if return_:
                allCanvas.append(canvas.detach().cpu())
                allMaskParams.append(maskParams.detach().cpu())
                allColors.append(colors.detach().cpu())
                allMasks.append(mask.detach().cpu());
                
            
        if return_:
            return {
                        'allCanvas': allCanvas,
                        'allMaskParams': allMaskParams,
                        'allColors': allColors,
                        'allMasks': allMasks,
                        'inputMaskGen': inp,
                    }
            
        return canvas
    
class resnetEncoder(nn.Module):
    def __init__(self, in_c, bnk):
        super(resnetEncoder, self).__init__()
        
        self.resnet = resnet18()
        self.resnet.conv1 = nn.Conv2d(in_c, 64, kernel_size=7, stride=2, padding=3)
        self.resnet.avgpool = nn.AdaptiveAvgPool2d(1)
        self.resnet.fc = nn.Sequential(
            nn.Linear(512, bnk),
            nn.BatchNorm1d(bnk),
            nn.ReLU(True)
        )
        
    def forward(self, x):
        return self.resnet(x)

def get_maskParam_fc(args):
    maskParam_fc = nn.Sequential(*[
        nn.Linear(args.bnk, args.nParam),
    ])
    return maskParam_fc

def get_color_fc(args):
    color_fc = nn.Sequential(*[
        nn.Linear(args.bnk, 3),
        nn.Sigmoid(),
    ])
    return color_fc

def initCanvas(size, bSz, device, mode='black', nc=3):
    if mode == 'black':
        return torch.zeros(bSz, nc, size, size, device=device)
    elif mode == 'rand':
        return torch.rand(bSz, nc, size, size, device=device)
    else:
        raise ValueError('Unknown canvas init mode')
        
def get_grid_coordinates(size, bSz, device):
    # -1, 1 or 0, 1
    x = torch.linspace(-1, 1, size).repeat(bSz, 1, size, 1).to(device)
    y = x.transpose(-1,-2)
    grid = torch.cat([x,y],1)
    return grid


class maskgen(nn.Module):
    """
        A pixelwise mask generation module with options for normalization, activation fct etc..
    """
    def __init__(self, in_channels, out_channels, hidden_size=128, use_GN=True, use_GN_before_sigmoid=True, use_relu=False, use_sinus_tanh=False):
        super().__init__()
        # options
        self.use_GN_before_sigmoid = use_GN_before_sigmoid and use_GN
        self.use_sinus_tanh = use_sinus_tanh
        
        # layers
        self.conv1 = nn.Conv2d(in_channels, hidden_size, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, kernel_size=1)
        self.conv4 = nn.Conv2d(hidden_size, out_channels, kernel_size=1)
        
        n_groups = 32
        self.norm1 = nn.GroupNorm(num_groups=n_groups, num_channels=hidden_size) if use_GN else identity()
        self.norm2 = nn.GroupNorm(num_groups=n_groups, num_channels=hidden_size) if use_GN else identity()
        self.norm3 = nn.GroupNorm(num_groups=n_groups, num_channels=hidden_size) if use_GN else identity()
        self.norm4 = nn.GroupNorm(num_groups=out_channels, num_channels=out_channels) if (use_GN and use_GN_before_sigmoid) else identity()

        self.actvn = actvn(use_relu=use_relu, use_sinus_tanh=use_sinus_tanh)

    def forward(self, x):
        # x is the input size
        x = self.norm1(self.conv1(x))
        x = self.actvn(x) #
        x = self.norm2(self.conv2(x))
        x = self.actvn(x) #
        x = self.norm3(self.conv3(x))
        x = self.actvn(x) #
        x = self.norm4(self.conv4(x))
        x = torch.sigmoid(x)
        return x
    
class identity(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x

class actvn(nn.Module):
    """
        An activation function that is either relu, tanh, or a split between tanh and sin.
        Default is tanh
    """
    def __init__(self, use_relu=False, use_sinus_tanh=False, sinus_scale=1.):
        super().__init__()
        #options
        self.use_relu = use_relu
        self.use_sinus_tanh = use_sinus_tanh
        self.sinus_scale = sinus_scale

    def forward(self, x):
        if self.use_relu:
            return F.relu(x)
        elif self.use_sinus_tanh:
            half_channel_size = x.size(1)//2
            x1, x2 = x[:,:half_channel_size,:,:], x[:,half_channel_size:,:,:]
            x = torch.cat([torch.tanh(x1), torch.sin(x2*self.sinus_scale)], 1)
            return x
        else:# default is tanh
            return torch.tanh(x)

    def __repr__(self):
        return 'use_relu: {}, use_sinus_tanh: {}, sinus_scale: {}'.format(self.use_relu, self.use_sinus_tanh, self.sinus_scale)