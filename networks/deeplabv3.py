import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from networks.decoder import build_decoder
from networks.backbone import build_backbone
from networks.encoder import build_Sencode
from networks.Drec import *
class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=7,
                 sync_bn=False, num_domain=3, freeze_bn=False,lam = 0.9):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.vencode = build_Sencode(0)
        self.mencode = build_Sencode(1)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.Drecoder = Adain(BatchNorm)

        if freeze_bn:
            self.freeze_bn()


    def forward(self, image, domain_code, imagemix, phase):
        v = None
        mixm = None
        md = None
        HDomain = None
        mixv = None
        if phase == 'train':
            v = self.vencode(domain_code)
            mixv = self.vencode(image)
            
            mixm = self.mencode(imagemix)
            md = self.mencode(domain_code)

            HDomain = self.Drecoder(v,md)
            # input = torch.cat([image, HDomain], dim=0)
            input = torch.cat([image, domain_code], dim=0)
        else:
            input = image

        x, low_level_feat = self.backbone(input)

        x = self.decoder(x,low_level_feat)

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x, v, mixm, md, HDomain, mixv



if __name__ == "__main__":
    model = DeepLab(backbone='resnet', output_stride=16)
    model.eval()
    image = torch.rand(1, 3, 512, 512)
    domain_code = torch.rand(1, 3, 512, 512)
    imagemix = torch.rand(1, 3, 512, 512)
    output, v, mixm, md,HDomain,mixv= model(image,domain_code,imagemix)

    print(md.size())


