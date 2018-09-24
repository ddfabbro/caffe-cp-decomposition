import caffe

def conv_layer(name, num_output, group, kernel_size, pad, stride):
    layer = caffe.proto.caffe_pb2.LayerParameter()
    layer.type = 'Convolution'
    layer.name = name
    layer.convolution_param.num_output = num_output
    layer.convolution_param.group = group
    
    if kernel_size[0] == kernel_size[1]:
        layer.convolution_param.kernel_size = kernel_size[0]
    else:
        layer.convolution_param.kernel_h = kernel_size[0]
        layer.convolution_param.kernel_w = kernel_size[1]
        
    if pad[0] == pad[1]:
        layer.convolution_param.pad = pad[0]
    else:   
        layer.convolution_param.pad_w = pad[0]
        layer.convolution_param.pad_h = pad[1]
    
    if stride[0] == stride[1]:
        layer.convolution_param.stride = stride[0]
    else:
        layer.convolution_param.stride_w = stride[0]
        layer.convolution_param.stride_h = stride[1]
        
    return layer
 
def decompose_layer(layer, rank):
    param = layer.convolution_param
    name = [layer.name+'_c', layer.name+'_y', layer.name+'_x', layer.name+'_n']
    
    num_output = param.num_output
    kernel_size = param.kernel_size
    
    pad = param.pad if hasattr(param, 'pad') else 0
    stride = param.stride if hasattr(param, 'stride') else 1
    
    decomposed_layer = [
        conv_layer(name[0], rank, 1, [1,1], [0,0], [1,1]),
        conv_layer(name[1], rank, rank, [kernel_size, 1], [pad,0], [stride,1]),
        conv_layer(name[2], rank, rank, [1, kernel_size], [0,pad], [1,stride]),
        conv_layer(name[3], num_output, 1, [1,1], pad=[0,0], stride=[1,1]),
    ]
    
    return decomposed_layer
