import os, sys
# from six.moves import urllib
import gzip
import errno
# import codecs
import numpy as np
import shutil
import PIL.Image as Image
import scipy.io as sio

import torch as tc
from torchvision.datasets.utils import download_url


urls = {
    'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
              "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
    'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
             "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
#     'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
#               "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]
    }


def download(root):
    # download files
    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    for k, v in urls.items():
        url = v[0]
        fn = v[1]
        md5 = v[2]
        print('Downloading ' + url)
        download_url(url, root, fn, md5)
        
def create_image_folder(root_tmp, root):
    
    tr_mat = sio.loadmat(os.path.join(root_tmp, "train_32x32.mat"))
    xs_tr = tr_mat['X']
    ys_tr = tr_mat['y'].astype(np.int64).squeeze()
    np.place(ys_tr, ys_tr == 10, 0)
    xs_tr = np.transpose(xs_tr, (3, 2, 0, 1))
    xs_tr = tc.tensor(xs_tr)
    ys_tr = tc.tensor(ys_tr)
    
    te_mat = sio.loadmat(os.path.join(root_tmp, "test_32x32.mat"))
    xs_te = te_mat['X']
    ys_te = te_mat['y'].astype(np.int64).squeeze()
    np.place(ys_te, ys_te == 10, 0)
    xs_te = np.transpose(xs_te, (3, 2, 0, 1))
    xs_te = tc.tensor(xs_te)
    ys_te = tc.tensor(ys_te)
    
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
            img = Image.fromarray(x.permute(1, 2, 0).numpy())
            img.save(os.path.join(dir_name, "%d.png"%(i+1)), "png")
            
if __name__ == "__main__":
    root = "SVHN"
    root_tmp = "SVHN_TMP"
    
    if os.path.exists(root):
        shutil.rmtree(root)
    download(root_tmp)
    create_image_folder(root_tmp, root)
    #if os.path.exists(root_tmp):
    #    shutil.rmtree(root_tmp)
    
