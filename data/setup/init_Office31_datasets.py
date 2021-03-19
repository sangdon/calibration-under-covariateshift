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
import glob
from shutil import copyfile, rmtree

url = "http://cis.upenn.edu/~sangdonp/data/OFFICE31_ori.tar"
    
def download(root):
    # download files
    try:
        os.makedirs(root)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    print('Downloading ' + url)
    data = urllib.request.urlopen(url)
    filename = url.rpartition('/')[2]
    file_path = os.path.join(root, filename)
    with open(file_path, 'wb') as f:
        f.write(data.read())
        
def create_image_folder(root_tmp_all, root_all):
    
    ## untar the tar file
    tar_fn = os.path.join(root_tmp_all, url.rpartition('/')[-1])
    os.system("tar -C %s -xvf %s"%(root_tmp_all, tar_fn))
    
    office31_dir = os.path.splitext(tar_fn)[0]
    subroot_dirs = [os.path.split(r)[1] for r in glob.glob(office31_dir+"/*")]

    ## for each dataset, split train/val/test
    for subroot in subroot_dirs:
        fns = [fn for fn in glob.glob(
            os.path.join(office31_dir, subroot, "**/*.jpg"), recursive=True)]
        print(len(fns))
        
        # randomly split tr/val/te files
        np.random.shuffle(fns)
        n = len(fns)
        tr_size = int(n*0.7)
        val_size = int(n*0.15)
        te_size = n - tr_size - val_size

        fns_all = fns
        fns_tr = fns[0:tr_size]
        fns_val = fns[tr_size:tr_size+val_size]
        fns_te = fns[tr_size+val_size:]

        ## write
        for dir_name_i, fns_i in [
            ('all', fns_all), ('train', fns_tr), ('val', fns_val), ('test', fns_te)]:
            for f in fns_i:
                src_fn = f
                dst_fn = os.path.join(
                    root_all, subroot, dir_name_i,
                    os.path.split(os.path.split(f)[0])[1], os.path.split(f)[1])
                
                if not os.path.exists(os.path.dirname(dst_fn)):
                    os.makedirs(os.path.dirname(dst_fn))
                print(dst_fn)
                copyfile(src_fn, dst_fn)
       

if __name__ == "__main__":
    root = "OFFICE31"
    root_tmp = "OFFICE31_TMP"
    
    if os.path.exists(root):
        shutil.rmtree(root)
    download(root_tmp)
    create_image_folder(root_tmp, root)
    if os.path.exists(root_tmp):
        shutil.rmtree(root_tmp)
    
"""
To do:
- create folders for all classes
"""