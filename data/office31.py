from torchvision import datasets, transforms
import torch as tc
import torch.utils.data as data
import sys, os
import types

tform_rnd = transforms.Compose(
    [transforms.RandomResizedCrop(224),
     transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     transforms.Normalize(
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225])])

tform_no_rnd = transforms.Compose(
    [transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize(
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225])])

def init_ld(root_dir, tform, batch_size, shuffle, num_workers):
    data = datasets.ImageFolder(root=root_dir, transform=tform)
    return tc.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, 
        drop_last=False, num_workers=num_workers)

def loadOffice(root_dir, dataset_name, batch_size, 
               train_shuffle=True, val_shuffle=True, test_shuffle=False):
    
    ld = types.SimpleNamespace()
    tform_tr = tform_rnd if train_shuffle else tform_no_rnd
    tform_val = tform_no_rnd
    tform_te = tform_no_rnd
    
    # train
    root_dir_tr = os.path.join(root_dir, dataset_name, 'train')
    ld.train = init_ld(root_dir_tr, tform_tr, batch_size, train_shuffle, 4)
    
    # val
    root_dir_val = os.path.join(root_dir, dataset_name, 'val')
    ld.val = init_ld(root_dir_val, tform_val, batch_size, val_shuffle, 2)
    
    # test
    root_dir_te = os.path.join(root_dir, dataset_name, 'test')
    ld.test = init_ld(root_dir_te, tform_te, batch_size, test_shuffle, 2)
    
    return ld


def loadAmazon(root_dir, batch_size, train_shuffle=True, val_shuffle=True, test_shuffle=False):
    return loadOffice(root_dir, "amazon", batch_size, train_shuffle, val_shuffle, test_shuffle)

def loadWebcam(root_dir, batch_size, train_shuffle=True, val_shuffle=True, test_shuffle=False):
    return loadOffice(root_dir, "webcam", batch_size, train_shuffle, val_shuffle, test_shuffle)

def loadDSLR(root_dir, batch_size, train_shuffle=True, val_shuffle=True, test_shuffle=False):
    return loadOffice(root_dir, "dslr", batch_size, train_shuffle, val_shuffle, test_shuffle)
