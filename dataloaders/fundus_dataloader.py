from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from scipy.stats import kurtosis
import glob
import random
import nibabel as nib
import random
import custom_transforms
class FundusSegmentation(Dataset):
    """
    Fundus segmentation dataset
    including 5 domain dataset
    one for test others for training
    """

    def __init__(self,
                 base_dir=None,
                 phase='train',
                 splitid=[1,2],
                 transform=None,
                 state='train',
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self.state = state
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase
        self.image_pool = {'HDB':[], 'WZE':[], 'SVI':[], 'TOP':[]}
        self.label_pool = {'HDB':[], 'WZE':[], 'SVI':[], 'TOP':[]}
        self.img_name_pool = {'HDB':[], 'WZE':[], 'SVI':[], 'TOP':[]}

        # self.image_pool = {'HDB':[], 'WZE':[], 'SVI':[]}
        # self.label_pool = {'HDB':[], 'WZE':[], 'SVI':[]}
        # self.img_name_pool = {'HDB':[], 'WZE':[], 'SVI':[]}

        self.flags_HDB = ['ze']
        self.flags_WZE = ['wz']
        self.flags_SVI = ['sv']
        self.flags_TOP = ['tc']

        # self.flags_HDB = ['Ci']
        # self.flags_WZE = ['Sp']
        # self.flags_SVI = ['To']


        self.splitid = splitid
        SEED = 1212
        random.seed(SEED)
        for id in splitid:
            self._image_dir = os.path.join(self._base_dir, 'Domain'+str(id), 'image')
            print('==> Loading {} data from: {}'.format(phase, self._image_dir))

            imagelist = glob.glob(self._image_dir + '/*')
            print(len(imagelist))
            for image_path in imagelist:
                gt_path = image_path.replace('image', 'mask')
                self.image_list.append({'image': image_path, 'label': gt_path})

        self.transform = transform
        self.transform1 = transforms.Compose([
            tr.f1(),
            tr.ToTensor()
        ])
        self.transform2 = transforms.Compose([
            tr.f2(),
            tr.ToTensor()
        ])
        self.transform3 = transforms.Compose([
            tr.f3(),
            tr.ToTensor()
        ])
        self._read_img_into_memory()
        for key in self.image_pool:
            if len(self.image_pool[key]) < 1:
                del self.image_pool[key]
                del self.label_pool[key]
                del self.img_name_pool[key]
                break
        for key in self.image_pool:
            if len(self.image_pool[key]) < 1:
                del self.image_pool[key]
                del self.label_pool[key]
                del self.img_name_pool[key]
                break
        for key in self.image_pool:
            if len(self.image_pool[key]) < 1:
                del self.image_pool[key]
                del self.label_pool[key]
                del self.img_name_pool[key]
                break
        # Display stats
        print('-----Total number of images in {}: {:d}'.format(phase, len(self.image_list)))

    def __len__(self):
        max = -1
        for key in self.image_pool:
             if len(self.image_pool[key])>max:
                 max = len(self.image_pool[key])
        return max

    def __getitem__(self, index):
        if self.phase != 'test':
            sample = []
            for key in self.image_pool:
                domain_code = list(self.image_pool.keys()).index(key)
                index = np.random.choice(len(self.image_pool[key]), 1)[0]
                _img = self.image_pool[key][index]
                _target = self.label_pool[key][index]
                _img_name = self.img_name_pool[key][index]
                anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name, 'dc': domain_code}
                downline = (1 / 3 + custom_transforms.f) * 255
                upline = (2 / 3 + custom_transforms.f) * 255
                pixel_array = np.array(_img.convert("L"), dtype=np.float64)
                flat_pixels = pixel_array.flatten()
                int1 = flat_pixels[(flat_pixels >= 0) & (flat_pixels < downline)]
                int2 = flat_pixels[(flat_pixels >= downline) & (flat_pixels < upline)]
                int3 = flat_pixels[(flat_pixels >= upline) & (flat_pixels <= 255)]
                kur1 = kurtosis(int1,fisher=True)
                kur2 = kurtosis(int2,fisher=True)
                kur3 = kurtosis(int3,fisher=True)
                if self.transform is not None:
                    if max(kur1,kur2,kur3) == kur1:
                        anco_sample = self.transform(anco_sample)
                        anco_sample = self.transform1(anco_sample)
                    elif max(kur1,kur2,kur3) == kur2:
                        anco_sample = self.transform(anco_sample)
                        anco_sample = self.transform2(anco_sample)
                    else:
                        anco_sample = self.transform(anco_sample)
                        anco_sample = self.transform3(anco_sample)
                sample.append(anco_sample)
        else:
            sample = []
            for key in self.image_pool:
                domain_code = list(self.image_pool.keys()).index(key)
                _img = self.image_pool[key][index]
                _target = self.label_pool[key][index]
                _img_name = self.img_name_pool[key][index]
                anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name, 'dc': domain_code}
                if self.transform is not None:
                    anco_sample = self.transform(anco_sample)
                sample=anco_sample
        # print(sample[0]['label'].size())
        # print(sample[0]['image'].size())
        return sample

    def _read_img_into_memory(self):
        img_num = len(self.image_list)
        for index in range(img_num):
            basename = os.path.basename(self.image_list[index]['image'])
            Flag = "NULL"
            if basename[0:2] in self.flags_HDB:
                Flag = 'HDB'
            elif basename[0:2] in self.flags_WZE:
                Flag = 'WZE'
            elif basename[0:2] in self.flags_SVI:
                Flag = 'SVI'
            elif basename[0:2] in self.flags_TOP:
                Flag = 'TOP'
            else:
                print("[ERROR:] Unknown dataset!")
                return 0
            arr1 = np.array(nib.load(self.image_list[index]['image']).get_fdata().astype(np.uint8))
            img = Image.fromarray(arr1).convert('RGB')
            img = img.resize((512, 512))

            # img = Image.open(self.image_list[index]['image']).convert('RGB')

            self.image_pool[Flag].append(img)

            _target = np.array(nib.load(self.image_list[index]['label']).get_fdata().astype(np.uint8))
            _target = np.squeeze(_target)
            _target = Image.fromarray(_target)
            _target = _target.resize((512, 512))

            # _target = Image.open(self.image_list[index]['label']).convert('L')

            # if _target.mode is 'RGB':
            #     _target = _target.convert('L')
            # if self.state != 'prediction':
            #     _target = _target.resize((256, 256))
            # print(_target.size)
            # print(_target.mode)
            self.label_pool[Flag].append(_target)
            # if self.split[0:4] in 'test':
            _img_name = self.image_list[index]['image'].split('/')[-1]
            self.img_name_pool[Flag].append(_img_name)

    def __str__(self):
        return 'Fundus(phase=' + self.phase+str(args.datasetTest[0]) + ')'

if __name__ == '__main__':
    import custom_transforms as tr
    from torch.utils.data import DataLoader
    from torchvision import transforms
    import matplotlib.pyplot as plt

    composed_transforms_tr = transforms.Compose([
        tr.RandomHorizontalFlip(),
        tr.RandomSized(512),
        tr.RandomRotate(15),
        tr.ToTensor()])

    voc_train = FundusSegmentation(split='train1',
                                   transform=composed_transforms_tr)

    dataloader = DataLoader(voc_train, batch_size=5, shuffle=True, num_workers=2)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            segmap = tmp
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0]).astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)

            break
    plt.show(block=True)
