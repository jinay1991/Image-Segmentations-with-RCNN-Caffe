
#
# Sample Dataset can be found at Cat-Dog Classification
# https://www.kaggle.com/c/dogs-vs-cats/download/train.zip
# https://www.kaggle.com/c/dogs-vs-cats/download/test1.zip
#
import cv2
import numpy as np
import os
import glob
import random
import lmdb
import caffe
from caffe.proto import caffe_pb2
import logging
from six.moves import urllib
import shutil

log = logging.getLogger()
logging.basicConfig(level=logging.INFO)



class CaffeRCNN():

    def __init__(self, caffe_root="/Users/jinay/workspace/git-repo/caffe"):
        self.dataset_dir = os.path.join(os.path.abspath(os.path.curdir), "data")
        self.train_lmdb = os.path.join(self.dataset_dir, "train_lmdb")
        self.validation_lmdb = os.path.join(self.dataset_dir, "validation_lmdb")
        self.model_dir = os.path.join(os.path.abspath(os.path.curdir), "models")

        self.caffe_root = caffe_root

        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if os.path.exists(self.train_lmdb):
            shutil.rmtree(self.train_lmdb)
        if os.path.exists(self.validation_lmdb):
            shutil.rmtree(self.validation_lmdb)


    def __transform_image(self, image, width=227, height=227):
        """
        Equalizes image color histograms and resize it to provided width x height
        """
        B, G, R = image[:,:,0], image[:,:,1], image[:,:,2]

        # histogram equalization for all the channels
        B = cv2.equalizeHist(B)
        G = cv2.equalizeHist(G)
        R = cv2.equalizeHist(R)

        # image resize
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

        return image

    def __make_datum(self, image, label, width=227, height=227):
        """
        """
        # image is np.ndarray format. BGR instead of RGB
        data = np.rollaxis(image, 2).tostring()
        return caffe_pb2.Datum(
            channels=3,
            width=width,
            height=height,
            label=label,
            data=data)

    def __create_lmdb(self, lmdb_file, list_of_image_data):
        """
        dumps lmdb_file for alls the image in data list
        """
        in_pb = lmdb.open(lmdb_file, map_size=int(1e12))
        with in_pb.begin(write=True) as in_txn:
            for in_idx, image_path in enumerate(list_of_image_data):
                if in_idx % 6 == 0:
                    continue
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                image = self.__transform_image(image)
                if "cat" in image_path:
                    label = 0
                else:
                    label = 1
                datum = self.__make_datum(image, label)
                in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
                log.info("{:0>5d}".format(in_idx) + ":" + image_path)
        in_pb.close()

    def create_train_lmdb(self, image_dir):
        """
        Creates train.lmdb for the training image files.
        """
        if not os.path.exists(self.train_lmdb):
            os.makedirs(self.train_lmdb)

        train_data = [image for image in glob.glob(image_dir + "/*.jpg")]

        random.shuffle(train_data)

        log.info("creating train_lmdb")

        self.__create_lmdb(self.train_lmdb, train_data)

    def create_validation_lmdb(self, image_dir):
        """
        Creates train.lmdb for the training image files.
        """
        if not os.path.exists(self.validation_lmdb):
            os.makedirs(self.validation_lmdb)

        test_data = [image for image in glob.glob(image_dir + "/*.jpg")]

        log.info("creating validation_lmdb")

        self.__create_lmdb(self.validation_lmdb, test_data)

    def get_model(self, existing_model=None):
        """
        Download Model from bvlc or symlink the existing one if set existing_model to path/to/model.caffemodel
        """

        rcnn_caffemodel_url = "http://dl.caffe.berkeleyvision.org/bvlc_reference_rcnn_ilsvrc13.caffemodel"

        if existing_model is None:
            self.caffemodel = os.path.join(self.model_dir, os.path.basename(rcnn_caffemodel_url))
            urllib.request.urlretrieve(rcnn_caffemodel_url, self.caffemodel)
        else:
            self.caffemodel = os.path.join(self.model_dir, os.path.basename(existing_model))

            if os.path.exists(self.caffemodel):
                os.remove(self.caffemodel)

            os.symlink(existing_model, self.caffemodel)

    def compute_mean_blob(self):
        """
        Computing Mean Image
        """
        # generate mean image
        self.compute_image_mean = os.path.join(self.caffe_root, "build", "tools", "compute_image_mean")
        self.mean_blob = os.path.join(self.dataset_dir, "mean.binaryproto")
        compute_image_mean_cmdline = str(self.compute_image_mean) + " -backend=lmdb " + str(self.train_lmdb) + " " + str(self.mean_blob)
        print compute_image_mean_cmdline
        log.info("running %s", compute_image_mean_cmdline)
        os.system(compute_image_mean_cmdline)

        mean_blob = caffe_pb2.BlobProto()
        with open(self.mean_blob) as f:
            mean_blob.ParseFromString(f.read())
        self.mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
            mean_blob.channels, mean_blob.height, mean_blob.width)

    def read_model(self, prototxt):
        
        caffe.set_mode_cpu()
        
        self.prototxt = os.path.abspath(prototxt)
        log.debug("prototxt: %s\ncaffemodel: %s", self.prototxt, self.caffemodel)
        self.net = caffe.Net(str(self.prototxt), str(self.caffemodel), caffe.TEST)

        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_mean('data', self.mean_array)
        self.transformer.set_transpose('data', (2, 0, 1))

    def test_image(self, image_file):
        """
        Run the network for the provided image file.
        """
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        image = self.__transform_image(image)

        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', image)
        out = self.net.forward()
        log.info(out)
        # pred_probas = out['prob']

        # log.info("Probability: ", pred_probas.argmax())

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input image file", required=True)
    parser.add_argument("--prototxt", help="input model prototxt", required=True)
    args = parser.parse_args()

    log.setLevel(level=logging.DEBUG)

    if not os.path.exists(os.path.abspath(args.input)):
        log.error("Incorrect File path for input file.")
        exit(1)

    rcnn = CaffeRCNN()
    rcnn.create_train_lmdb(image_dir=os.path.join(os.path.abspath(os.path.curdir), "data", "train"))
    rcnn.create_validation_lmdb(image_dir=os.path.join(os.path.abspath(os.path.curdir), "data", "test1"))

    rcnn.get_model(existing_model="/Users/jinay/workspace/git-repo/caffe/models/bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel")

    rcnn.compute_mean_blob()

    rcnn.read_model(args.prototxt)

    rcnn.test_image(args.input)