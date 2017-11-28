"""
Modification of
- https://github.com/carpedm20/DCGAN-tensorflow/blob/master/download.py
- http://stackoverflow.com/a/39225039
- https://github.com/shekkizh/WassersteinGAN.tensorflow/blob/master/Dataset_Reader/read_celebADataset.py
"""
from __future__ import print_function
import os
import zipfile
import requests
import subprocess
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
import sys, inspect
import random
from six.moves import cPickle as pickle
from tensorflow.python.platform import gfile
import glob
from PIL import Image
import scipy.io
import scipy.misc as misc

#utils_path = os.path.abspath(
#    os.path.realpath(os.path.join(os.path.split(inspect.getfile(inspect.currentframe()))[0], "..")))
#if utils_path not in sys.path:
#   sys.path.insert(0, utils_path)
#import utils as utils

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={ 'id': id }, stream=True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination, chunk_size=32*1024):
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size), total=total_size, 
                          unit='B', unit_scale=True, desc=destination):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def unzip(filepath):
    print("Extracting: " + filepath)
    base_path = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(base_path)
    os.remove(filepath)

def download_celeb_a(base_path):
    #data_path = os.path.join(base_path, 'CelebA')
    #images_path = os.path.join(data_path, 'images')
    #if os.path.exists(data_path):
    #    print('[!] Found Celeb-A - skip')
    #    return

    filename, drive_id  = "img_align_celeba.zip", "0B7EVK8r0v71pZjFTYXZWM3FlRnM"
    save_path = os.path.join(base_path, filename)

    if os.path.exists(save_path):
        print('[*] {} already exists'.format(save_path))
    else:
        download_file_from_google_drive(drive_id, save_path)

    zip_dir = ''
    with zipfile.ZipFile(save_path) as zf:
        zip_dir = zf.namelist()[0]
        zf.extractall(base_path)
    #if not os.path.exists(data_path):
    #    os.mkdir(data_path)
    #os.rename(os.path.join(base_path, "img_align_celeba"), images_path)
    #os.remove(save_path)

class CelebA_Dataset():
    def __init__(self, dict):
        self.train_images = dict['train']
        self.test_images = dict['test']
        self.validation_images = dict['validation']

def Pickle_Dataset(base_path):
    pickle_filename = "celebA.pickle"
    pickle_filepath = os.path.join(base_path, pickle_filename)

    #celebA_folder = os.path.splitext(DATA_URL.split("/")[-1])[0]
    dir_path = os.path.join(base_path, "img_align_celeba")
    #dir_path = os.path.join(base_path, "images3")

    result = create_image_lists(dir_path)

    print ("Training set: %d" % len(result['train']))
    print ("Test set: %d" % len(result['test']))
    print ("Validation set: %d" % len(result['validation']))
    print ("Pickling ...")
    with open(pickle_filepath, 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

    print("Pickle finished into ", pickle_filename)

    '''
    if not os.path.exists(pickle_filepath):
        # utils.maybe_download_and_extract(data_dir, DATA_URL, is_zipfile=True)
        #celebA_folder = os.path.splitext(DATA_URL.split("/")[-1])[0]
        #dir_path = os.path.join(data_dir, img_align_celeba)
        if not os.path.exists(dir_path):
            print ("CelebA dataset needs to be downloaded and unzipped manually")
            print ("Download from: %s" % DATA_URL)
            raise ValueError("Dataset not found")

        result = create_image_lists(dir_path)
        print ("Training set: %d" % len(result['train']))
        print ("Test set: %d" % len(result['test']))
        print ("Validation set: %d" % len(result['validation']))
        print ("Pickling ...")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)
    else:
        print ("Found pickle file!")

    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        celebA = CelebA_Dataset(result)
        del result
    return celebA
    '''


def create_image_lists(image_dir, testing_percentage=0.01, validation_percentage=0.01):
    """
    Code modified from tensorflow/tensorflow/examples/image_retraining
    """
    if not gfile.Exists(image_dir):
        print("Image directory '" + image_dir + "' not found.")
        return None
    training_images = []
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    sub_dirs = [x[0] for x in os.walk(image_dir)]
    file_list = []

    for extension in extensions:
        file_glob = os.path.join(image_dir, '*.' + extension)
        file_list.extend(glob.glob(file_glob))

    if not file_list:
        print('No files found')
    else:
        # print "No. of files found: %d" % len(file_list)
        training_images.extend([f for f in file_list])

    random.shuffle(training_images)
    no_of_images = len(training_images)
    validation_offset = int(validation_percentage * no_of_images)
    validation_images = training_images[:validation_offset]
    test_offset = int(testing_percentage * no_of_images)
    testing_images = training_images[validation_offset:validation_offset + test_offset]
    training_images = training_images[validation_offset + test_offset:]

    result = {
        'train': training_images,
        'test': testing_images,
        'validation': validation_images,
    }
    return result

'''
#def prepare_data_dir(path = './data'):
#    if not os.path.exists(path):
#        os.mkdir(path)

# check, if file exists, make link
#def check_link(in_dir, basename, out_dir):
#    in_file = os.path.join(in_dir, basename)
#    if os.path.exists(in_file):
#        link_file = os.path.join(out_dir, basename)
        rel_link = os.path.relpath(in_file, out_dir)
        os.symlink(rel_link, link_file)

def add_splits(base_path):
    data_path = os.path.join(base_path, 'CelebA')
    images_path = os.path.join(data_path, 'images')
    train_dir = os.path.join(data_path, 'splits', 'train')
    valid_dir = os.path.join(data_path, 'splits', 'valid')
    test_dir = os.path.join(data_path, 'splits', 'test')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(valid_dir):
        os.makedirs(valid_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # these constants based on the standard CelebA splits
    NUM_EXAMPLES = 202599
    TRAIN_STOP = 162770
    VALID_STOP = 182637

    for i in range(0, TRAIN_STOP):
        basename = "{:06d}.jpg".format(i+1)
        check_link(images_path, basename, train_dir)
    for i in range(TRAIN_STOP, VALID_STOP):
        basename = "{:06d}.jpg".format(i+1)
        check_link(images_path, basename, valid_dir)
    for i in range(VALID_STOP, NUM_EXAMPLES):
        basename = "{:06d}.jpg".format(i+1)
        check_link(images_path, basename, test_dir)

'''

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def save_image_RGB( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "RGB" )
    img.save( outfilename )

def resize_images(base_path, resize):
    celebA_path = os.path.join(base_path, "img_align_celeba")
    filelist = glob.glob(celebA_path + '/*.jpg')
    #print(filelist)
    for fname in filelist:
        image = load_image(fname)
        img_resized = misc.imresize(image, (resize, resize))
        #print(img_resized)
        save_image_RGB(img_resized,fname)


if __name__ == '__main__':
    base_path = '.'
    #prepare_data_dir()
    download_celeb_a(base_path)
    resize_images(base_path, 128)
    Pickle_Dataset(base_path)
    #add_splits(base_path)
