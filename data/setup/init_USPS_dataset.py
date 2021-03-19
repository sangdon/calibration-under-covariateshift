import os, sys
from six.moves import urllib
import h5py

# import gzip
import errno
# import codecs
import numpy as np
import torch as tc
import shutil
import PIL.Image as Image


urls = ["http://cis.upenn.edu/~sangdonp/data/usps.h5"]
    
def download(root):
    # download files
    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    for url in urls:
        print('Downloading ' + url)
        data = urllib.request.urlopen(url)
        filename = url.rpartition('/')[2]
        file_path = os.path.join(root, filename)
        with open(file_path, 'wb') as f:
            f.write(data.read())

def create_image_folder(root_tmp, root):
    
    with h5py.File(os.path.join(root_tmp, "usps.h5"), 'r') as hf:
        train = hf.get('train')
        xs_tr = tc.tensor(train.get('data')).view(-1, 16, 16)
        xs_tr = (xs_tr * 255.0).byte()
        ys_tr = tc.tensor(train.get('target'))
        test = hf.get('test')
        xs_te = tc.tensor(test.get('data')[:]).view(-1, 16, 16)
        xs_te = (xs_te * 255.0).byte()
        ys_te = tc.tensor(test.get('target')[:])
        
    print(xs_tr.size())
    print(ys_tr.size())
    print(xs_te.size())
    print(ys_te.size())
    
    n_val = int(xs_tr.size(0) * 0.15)
    n_tr = xs_tr.size(0) - n_val
    xs_val = xs_tr[n_tr:]
    ys_val = ys_tr[n_tr:]
    xs_tr = xs_tr[:n_tr]
    ys_tr = ys_tr [:n_tr]
    
    
    print(xs_tr.size())
    print(ys_tr.size())
    print(xs_val.size())
    print(ys_val.size())
    print(xs_te.size())
    print(ys_te.size())
    
    for mode in ["train", "val", "test", "all"]:
        if not os.path.exists(os.path.join(root, mode)):
            os.makedirs(os.path.join(root, mode))
        if mode == "train":
            xs = xs_tr
            ys = ys_tr
        elif mode == "val":
            xs = xs_val
            ys = ys_val
        elif mode == "test":
            xs = xs_te
            ys = ys_te
        elif mode == "all":
            xs = tc.cat((xs_tr, xs_val, xs_te), 0)
            ys = tc.cat((ys_tr, ys_val, ys_te), 0)
            
        else:
            raise NotImplementedError
        # write images
        for i, (x, y) in enumerate(zip(xs, ys)):
            dir_name = os.path.join(root, mode, str(y.item()))
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            img = Image.fromarray(x.numpy(), mode='L')
            img.save(os.path.join(dir_name, "%d.png"%(i+1)), "png")
            

if __name__ == "__main__":
    root = "USPS"
    root_tmp = "USPS_TMP"
    
    if os.path.exists(root):
        shutil.rmtree(root)
    download(root_tmp)
    create_image_folder(root_tmp, root)
    if os.path.exists(root_tmp):
        shutil.rmtree(root_tmp)
    