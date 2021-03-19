import os
from six.moves import urllib
import gzip
import errno
import codecs
import numpy as np
import torch as tc
import shutil
import PIL.Image as Image

urls = [
    'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
    'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
]

def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return tc.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return tc.from_numpy(parsed).view(length, num_rows, num_cols)

    
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
        with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                gzip.GzipFile(file_path) as zip_f:
            out_f.write(zip_f.read())
        os.unlink(file_path)

def create_image_folder(root_tmp, root):
    training_set = (
        read_image_file(os.path.join(root_tmp, 'train-images-idx3-ubyte')),
        read_label_file(os.path.join(root_tmp, 'train-labels-idx1-ubyte'))
    )
    test_set = (
        read_image_file(os.path.join(root_tmp, 't10k-images-idx3-ubyte')),
        read_label_file(os.path.join(root_tmp, 't10k-labels-idx1-ubyte'))
    )
    
    xs_tr, ys_tr = training_set
    xs_val = xs_tr[50000:]
    ys_val = ys_tr[50000:]
    xs_tr = xs_tr[:50000]
    ys_tr = ys_tr [:50000]
    xs_te, ys_te = test_set
    
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
    root = "MNIST"
    root_tmp = "MNIST_TMP"
    
    if os.path.exists(root):
        shutil.rmtree(root)
    download(root_tmp)
    create_image_folder(root_tmp, root)
    if os.path.exists(root_tmp):
        shutil.rmtree(root_tmp)
    