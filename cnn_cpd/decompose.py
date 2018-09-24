import numpy as np
from sktensor import dtensor, cp_als
import google.protobuf.text_format
from cnn_cpd.utils import caffe, decompose_layer

def decompose_model(model_def_path, model_weights_path, layer_ranks):

    """ CREATE DECOMPOSED MODEL DEFINITION """
    
    with open(model_def_path) as f:
        model_def = caffe.proto.caffe_pb2.NetParameter()
        google.protobuf.text_format.Merge(f.read(), model_def)
    
    new_model_def = caffe.proto.caffe_pb2.NetParameter()
    new_model_def.name = model_def.name + '_decomposed'
    
    if model_def.input:
        new_model_def.input.extend(['data'])
        new_model_def.input_dim.extend(model_def.input_dim)
    
    new_layers = [] #Keeping track of new layers helps renaming nodes in the future

    for layer in model_def.layer:
        if layer.name not in layer_ranks.keys() or layer.type != 'Convolution':
            new_model_def.layer.extend([layer])
        else:
            decomposed_layer = decompose_layer(layer, layer_ranks[layer.name])
            for i in range(4):
                new_layers.append(decomposed_layer[i].name)
            new_model_def.layer.extend(decomposed_layer)
    
    #Rename bottom/top nodes for some layers !!!
    layer_index = len(new_model_def.layer)
    for i in range(layer_index):
        #Label Decomposed layers nodes
        if new_model_def.layer[i].name in new_layers:
            if i == 0:
                new_model_def.layer[i].bottom.extend(['data'])
            elif new_model_def.layer[i-1].type == 'ReLU':
                new_model_def.layer[i].bottom.extend([new_model_def.layer[i-2].name])
            elif new_model_def.layer[i-1].type in ['Convolution','Pooling']:
                new_model_def.layer[i].bottom.extend([new_model_def.layer[i-1].name])
            new_model_def.layer[i].top.extend([new_model_def.layer[i].name])
        #Rename Convolution layers nodes
        elif new_model_def.layer[i].type == 'Convolution':
            if new_model_def.layer[i-2].name in new_layers:
                new_model_def.layer[i].bottom[0] = new_model_def.layer[i-2].name
        #Rename ReLU layers nodes
        elif new_model_def.layer[i].type == 'ReLU':
            if new_model_def.layer[i-1].name in new_layers:
                new_model_def.layer[i].bottom[0] = new_model_def.layer[i-1].name
                new_model_def.layer[i].top[0] = new_model_def.layer[i-1].name
        #Rename Pooling layers nodes
        elif new_model_def.layer[i].type == 'Pooling':
            if new_model_def.layer[i-2].name in new_layers:
                new_model_def.layer[i].bottom[0] = new_model_def.layer[i-2].name

    new_model_def_path = model_def_path[:-9] + '_decomposed.prototxt'
    with open(new_model_def_path, 'w') as f:
        google.protobuf.text_format.PrintMessage(new_model_def, f)
        
    """ CREATE DECOMPOSED MODEL WEIGHTS """
    
    net = caffe.Net(model_def_path, model_weights_path, caffe.TEST)
    decomposed_net = caffe.Net(new_model_def_path, model_weights_path, caffe.TEST)
    
    layers = net.params.items()
    net_dict = {}
    
    for name, params in layers:
        net_dict[name] = {
            'weights': params[0].data, 
            'bias': params[1].data,
        }
      
    log_dict = {} #Log containing fit, n_itr and exectimes for each layer
    
    for conv_layer in layer_ranks.keys():
        log_dict[conv_layer] = {}
        rank = layer_ranks[conv_layer]
        T = dtensor(net_dict[conv_layer]['weights'])
        print('\nDecomposing %s...' %conv_layer)
        P, fit, n_itr, exectimes = cp_als(T, rank, init='random')
        print('Reconstruction: %f%%' %(fit*100))
        print('Elapsed time: %.2fs' %sum(exectimes))
        
        log_dict[conv_layer]['fit'] = fit[0]
        log_dict[conv_layer]['n_itr'] = n_itr
        log_dict[conv_layer]['exectime'] = sum(exectimes)
        
        num_output = net_dict[conv_layer]['weights'].shape[0]
        channels = net_dict[conv_layer]['weights'].shape[1]
        kernel_size = net_dict[conv_layer]['weights'].shape[2]
        bias = net_dict[conv_layer]['bias']
        
        P_x = (P.U[3]*P.lmbda).T
        P_y = P.U[2].T
        P_c = P.U[1].T
        P_n = P.U[0]
        
        P_x = np.reshape(P_x, [rank, 1, 1, kernel_size]).astype(np.float32)
        P_y = np.reshape(P_y, [rank, 1, kernel_size, 1]).astype(np.float32)
        P_c = np.reshape(P_c, [rank, channels, 1, 1]).astype(np.float32)
        P_n = np.reshape(P_n, [num_output, rank, 1, 1]).astype(np.float32)
        
        np.copyto(decomposed_net.params[conv_layer+'_x'][0].data, P_x)
        np.copyto(decomposed_net.params[conv_layer+'_y'][0].data, P_y)
        np.copyto(decomposed_net.params[conv_layer+'_c'][0].data, P_c)
        np.copyto(decomposed_net.params[conv_layer+'_n'][0].data, P_n)
        np.copyto(decomposed_net.params[conv_layer+'_n'][1].data, bias)
        
    new_model_weights_path = model_weights_path[:-11] + '_decomposed.caffemodel'
    decomposed_net.save(new_model_weights_path)
        
    return log_dict, [new_model_def_path, new_model_weights_path]
