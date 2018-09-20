import cnn_cpd as cpd
import urllib
import os

ROOT_DIR = '/path/to/cp_decomposition/'

if not os.path.isfile('models/VGG_ILSVRC_16_layers_deploy.caffemodel'):
    caffemodel_url = 'http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel'
    urllib.urlretrieve (caffemodel_url, 'models/VGG_ILSVRC_16_layers_deploy.caffemodel')

model = {
    'def': os.path.join(ROOT_DIR,'models/VGG_ILSVRC_16_layers_deploy.prototxt'),
    'weights': os.path.join(ROOT_DIR,'models/VGG_ILSVRC_16_layers_deploy.caffemodel'),
}

layer_ranks = {
    'conv3_1': 5,
    'conv3_2': 5,
    'conv3_3': 5,
    'conv4_1': 5,
    'conv4_2': 5,
    'conv4_3': 5,
}

new_model_def_path = cpd.decompose_model_def(model['def'], layer_ranks)
cpd.decompose_model_weights(model['def'], model['weights'], new_model_def_path, layer_ranks)
