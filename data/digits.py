import types
import os, sys

import torch as tc
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset

def loadMNIST(
    root, batch_size, image_size=28, gray=True, load_memory=False, tform_aug=None, 
    train_shuffle=True, val_shuffle=True, test_shuffle=False):
    
    mnist_root = root
    tforms = []
    # gray scaling
    if gray:
        tforms.append(transforms.Grayscale(1))
    else:
        tforms.append(transforms.Grayscale(3))
    # resize
    tforms.append(transforms.Resize([image_size, image_size]))
    # data augmetation
    if tform_aug is not None:
#         tform_aug = transforms.RandomAffine(degrees=15, scale=(0.8, 1.2))
        tforms.append(tform_aug)
    # tform to tensor
    tforms.append(transforms.ToTensor())
        
    ld = types.SimpleNamespace()
    # train
    if load_memory:
        ld.train = DataLoader(
            datasets.ImageFolder(
                os.path.join(mnist_root, "train"), transform=transforms.Compose(tforms)), 
            batch_size=batch_size, shuffle=False, num_workers=0)
        xs_tr = tc.cat([x for x, _ in ld.train], 0)
        ys_tr = tc.cat([y for _, y in ld.train], 0)
        ld.train = DataLoader(
            TensorDataset(xs_tr, ys_tr), batch_size=batch_size, shuffle=train_shuffle, num_workers=4)
    else:        
        ld.train = DataLoader(
            datasets.ImageFolder(
                os.path.join(mnist_root, "train"), transform=transforms.Compose(tforms)), 
            batch_size=batch_size, shuffle=train_shuffle, num_workers=4)

    # val
    if load_memory:        
        ld.val = tc.utils.data.DataLoader(
            datasets.ImageFolder(
                os.path.join(mnist_root, "val"), transform=transforms.Compose(tforms)), 
            batch_size=batch_size, shuffle=False, num_workers=0)
        xs_val = tc.cat([x for x, _ in ld.val], 0)
        ys_val = tc.cat([y for _, y in ld.val], 0)
        ld.val = DataLoader(
            TensorDataset(xs_val, ys_val), batch_size=batch_size, shuffle=val_shuffle, num_workers=4)
    else:
        ld.val = tc.utils.data.DataLoader(
            datasets.ImageFolder(
                os.path.join(mnist_root, "val"), transform=transforms.Compose(tforms)), 
            batch_size=batch_size, shuffle=val_shuffle, num_workers=4)
        
    # test    
    if load_memory:
        ld.test = tc.utils.data.DataLoader(
            datasets.ImageFolder(
                os.path.join(mnist_root, "test"), transform=transforms.Compose(tforms)), 
            batch_size=batch_size, shuffle=False, num_workers=0)
        xs_te = tc.cat([x for x, _ in ld.test], 0)
        ys_te = tc.cat([y for _, y in ld.test], 0)
        ld.test = DataLoader(
            TensorDataset(xs_te, ys_te), batch_size=batch_size, shuffle=test_shuffle, num_workers=0)
    else:
        ld.test = tc.utils.data.DataLoader(
            datasets.ImageFolder(
                os.path.join(mnist_root, "test"), transform=transforms.Compose(tforms)), 
            batch_size=batch_size, shuffle=test_shuffle, num_workers=0)
    return ld

def loadUSPS(
    root, batch_size, image_size=16, gray=True, tform_aug=None, 
    train_shuffle=True, val_shuffle=True, test_shuffle=False):
    usps_root = root
    tforms = []
    if gray:
        tforms.append(transforms.Grayscale(1))
    else:
        tforms.append(transforms.Grayscale(3))
    tforms.append(transforms.Resize([image_size, image_size]))
    # data augmetation
    if tform_aug is not None:
#         tform_aug = transforms.RandomAffine(degrees=15, scale=(0.8, 1.2))
        tforms.append(tform_aug)
    # tform to tensor
    tforms.append(transforms.ToTensor())
    
    ld = types.SimpleNamespace()
    ld.train = tc.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(usps_root, "train"), transform=transforms.Compose(tforms)), 
        batch_size=batch_size, shuffle=train_shuffle, num_workers=4)
    ld.val = tc.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(usps_root, "val"), transform=transforms.Compose(tforms)), 
        batch_size=batch_size, shuffle=val_shuffle, num_workers=4)
    ld.test = tc.utils.data.DataLoader(
        datasets.ImageFolder(os.path.join(usps_root, "test"), transform=transforms.Compose(tforms)), 
        batch_size=batch_size, shuffle=test_shuffle, num_workers=0)
    return ld

def loadSVHN(
    root, batch_size, image_size=32, gray=False, load_memory=False, tform_aug=None, 
    train_shuffle=True, val_shuffle=True, test_shuffle=False):
    svhn_root = root
    tforms = []
    if gray:
        tforms.append(transforms.Grayscale())
    tforms.append(transforms.Resize([image_size, image_size]))
    # data augmetation
    if tform_aug is not None:
#         tform_aug = transforms.RandomAffine(degrees=15, scale=(0.8, 1.2))
        tforms.append(tform_aug)
    # tform to tensor
    tforms.append(transforms.ToTensor())
    
    ld = types.SimpleNamespace()
    if load_memory:
        ld.train = tc.utils.data.DataLoader(
            datasets.ImageFolder(
                os.path.join(svhn_root, "train"), transform=transforms.Compose(tforms)), 
            batch_size=batch_size, shuffle=False, num_workers=0)
        ld.val = tc.utils.data.DataLoader(
            datasets.ImageFolder(
                os.path.join(svhn_root, "val"), transform=transforms.Compose(tforms)), 
            batch_size=batch_size, shuffle=False, num_workers=0)
        ld.test = tc.utils.data.DataLoader(
            datasets.ImageFolder(
                os.path.join(svhn_root, "test"), transform=transforms.Compose(tforms)), 
            batch_size=batch_size, shuffle=False, num_workers=0)
        
        xs_tr = tc.cat([x for x, _ in ld.train], 0)
        ys_tr = tc.cat([y for _, y in ld.train], 0)
        ld.train = DataLoader(
            TensorDataset(xs_tr, ys_tr), batch_size=batch_size, shuffle=train_shuffle, num_workers=0)
        
        xs_val = tc.cat([x for x, _ in ld.val], 0)
        ys_val = tc.cat([y for _, y in ld.val], 0)
        ld.val = DataLoader(
            TensorDataset(xs_val, ys_val), batch_size=batch_size, shuffle=val_shuffle, num_workers=0)
        
        xs_te = tc.cat([x for x, _ in ld.test], 0)
        ys_te = tc.cat([y for _, y in ld.test], 0)
        ld.test = DataLoader(
            TensorDataset(xs_te, ys_te), batch_size=batch_size, shuffle=test_shuffle, num_workers=0)
    else:
        ld.train = tc.utils.data.DataLoader(
            datasets.ImageFolder(
                os.path.join(svhn_root, "train"), transform=transforms.Compose(tforms)), 
            batch_size=batch_size, shuffle=train_shuffle, num_workers=4)
        ld.val = tc.utils.data.DataLoader(
            datasets.ImageFolder(
                os.path.join(svhn_root, "val"), transform=transforms.Compose(tforms)), 
            batch_size=batch_size, shuffle=val_shuffle, num_workers=4)
        ld.test = tc.utils.data.DataLoader(
            datasets.ImageFolder(
                os.path.join(svhn_root, "test"), transform=transforms.Compose(tforms)), 
            batch_size=batch_size, shuffle=test_shuffle, num_workers=0)
    return ld