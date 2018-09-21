import os
import urllib
from collections import OrderedDict
import cnn_cpd as cpd

ROOT_DIR = '/path/to/CNN_CPD/'
os.chdir(ROOT_DIR)

if not os.path.isfile('models/VGG_ILSVRC_16_layers_deploy.caffemodel'):
    caffemodel_url = 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel'
    urllib.urlretrieve (caffemodel_url, 'models/VGG_ILSVRC_16_layers_deploy.caffemodel')

model = {
    'def': 'models/VGG_ILSVRC_16_layers_deploy.prototxt',
    'weights': 'models/VGG_ILSVRC_16_layers_deploy.caffemodel',
}

layer_ranks = OrderedDict([
    ('conv4_1', 175),
    ('conv4_2', 192),
    ('conv4_3', 227),
    ('conv5_1', 398),
    ('conv5_2', 390),
    ('conv5_3', 379),
])

cpd_data, _ = cpd.decompose_model(model['def'], model['weights'], layer_ranks)
