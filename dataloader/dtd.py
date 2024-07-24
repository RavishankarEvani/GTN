'''
Code for this DataLoader is sourced from: https://github.com/jiaxue-ai/pytorch-material-classification/tree/master/dataloader
This repository also contains DataLoaders for other datasets. 
'''

import os
import os.path

import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(txtnames, datadir, class_to_idx):
    images = []
    labels = []
    for txtname in txtnames:
        with open(txtname, 'r') as lines:
            for line in lines:
                classname = line.split('/')[0]
                _img = os.path.join(datadir, 'images', line.strip())
                assert os.path.isfile(_img)
                images.append(_img)
                labels.append(class_to_idx[classname])

    return images, labels


class DTDDataloader(data.Dataset):
    def __init__(self, cfg, transform=None, train=True):
        
        """
         Args:
             cfg (dict): config file.
             transform (callable, optional): Optional transform to be applied
                 on a sample.
            train (bool): Whether to train or test 
        """
        
        classes, class_to_idx = find_classes(os.path.join(cfg.paths.data, 'images'))
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.train = train
        self.transform = transform
        self.cfg = cfg

        if train:
            filename = [os.path.join(cfg.paths.data, 'labels/train' + str(cfg.training.split) + '.txt'),
                        os.path.join(cfg.paths.data, 'labels/val' + str(cfg.training.split) + '.txt')]
        else:
            filename = [os.path.join(cfg.paths.data, 'labels/test' + str(cfg.training.split) + '.txt')]

        self.images, self.labels = make_dataset(filename, cfg.paths.data, class_to_idx)
        assert (len(self.images) == len(self.labels))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        _img = Image.open(self.images[idx]).convert('RGB') #L
            
        _label = self.labels[idx]
        if self.transform is not None:
            _img = self.transform(_img)

        return _img, _label



class Dataloader():
    def __init__(self, cfg):

        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
        
            
        if cfg.common.additional_augmentation == True:
            
            transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAutocontrast(),
                transforms.ColorJitter(0.4,0.4,0.4),
                transforms.ToTensor(),
                normalize,
            ])
        
        else:
            
                transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
                
                
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
            
        

        trainset = DTDDataloader(cfg, transform_train, train=True)
        testset = DTDDataloader(cfg, transform_test, train=False)

        kwargs = {'num_workers': 8, 'pin_memory': True}
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=
        cfg.training.batch_size, shuffle=True, **kwargs)
        testloader = torch.utils.data.DataLoader(testset, batch_size=
        cfg.training.batch_size, shuffle=False, **kwargs)
        self.classes = trainset.classes
        self.trainloader = trainloader
        self.testloader = testloader

    def getloader(self):
        return self.classes, self.trainloader, self.testloader