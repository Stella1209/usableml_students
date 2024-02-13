import torch.nn as nn
import torch.nn.functional as F
from itertools import chain


class Adjustable_model(nn.Module):
    
    def __init__(self, linear_layers=[], convolutional_layers = [], output_classes = 10, input_size = (28,28)):
        super(Adjustable_model, self).__init__()
        
        output_size = input_size
        input_size = input_size[0]*input_size[1]
        
        self.conv_layers = convolutional_layers
        self.lin_layers = linear_layers
        self.convolutional_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        
        #self.forward_counter = 0
        
        self.input_size = input_size
        
        
        if self.conv_layers != []:
            input_size = 1
            for params in self.conv_layers: 
                if "kernel_size" not in params:
                    params['kernel_size'] = 3
                if "stride" not in params:
                    params['stride'] = 1
                if "padding" not in params:
                    params['padding'] = 0
                if "pooling" not in params:
                    params['pooling'] = "Off"
                if "output_function" not in params:
                    params['output_function'] = "Tanh"
                
                self.convolutional_layers.append(nn.Conv2d(in_channels = input_size, out_channels = params['size'], 
                                                           kernel_size=params['kernel_size'], stride=params['stride'], padding=params['padding']))
                input_size = params['size']
                #output_size = (output_size[0] - (params['kernel_size']-1), output_size[1] - (params['kernel_size']-1))
                output_size = ( int((output_size[0] - (params['kernel_size']) + 2*params['padding'])/params['stride'] +1) , int((output_size[1] - (params['kernel_size']) + 2*params['padding'])/params['stride'] +1))

                if params["pooling"] != "Off":
                    pool_kernel = int(params["pooling"])
                    self.convolutional_layers.append(nn.MaxPool2d(kernel_size=pool_kernel, stride=1))
                    output_size = ( int(output_size[0] - pool_kernel + 1) , int(output_size[1] - pool_kernel +1))
                    
                    
            input_size = input_size*output_size[0]*output_size[1]
            
            _conv = self.conv_layers        
            self.conv_layers = list(chain.from_iterable((i, {'output_function':'Pooling_layer'}) if i['pooling'] != 'Off' else [i] for i in _conv))
        
        
        for layer in self.lin_layers:
            if "output_function" not in layer:
                    layer['output_function'] = "Tanh"
            size = layer['linear_cells']
            self.linear_layers.append(nn.Linear(input_size, size)) # Linear layers only need layer size as a parameter which is i
            input_size = size
        
        # output layer:
        self.final_layer = nn.Linear(input_size, output_classes)   
    
    def forward(self, inputs):
        
        if len(self.convolutional_layers) != 0:
            for index, conv_layer in enumerate(self.convolutional_layers):
                if self.conv_layers[index]["output_function"] == "Tanh":
                    inputs = F.tanh(conv_layer(inputs))
                    #if self.forward_counter % 100 == 0:
                    #    print("Convolutional - Tanh")
                elif self.conv_layers[index]["output_function"] == "Softmax":
                    inputs = F.softmax(conv_layer(inputs))
                    #if self.forward_counter % 100 == 0:
                    #    print("Convolutional - Softmax")
                elif self.conv_layers[index]["output_function"] == "ReLu":
                    inputs = F.relu(conv_layer(inputs))
                    #if self.forward_counter % 100 == 0:
                    #    print("Convolutional - ReLu")
                elif self.conv_layers[index]["output_function"] == "Pooling_layer":
                    inputs = conv_layer(inputs)
                
                #if self.conv_layers[index]["pooling"] != "Off":
                #    pool_kernel = int(self.conv_layers[index]["pooling"])
                #    nn.MaxPool2d(kernel_size=pool_kernel, stride=1)
                #if self.forward_counter % 100 == 0:
                #    print(f"Max pooling - Kernel size: {pool_kernel}")
                    
                    
        
        #inputs = torch.flatten(inputs, 1)
        inputs = inputs.view(inputs.size(0), -1)
        for index, layer in enumerate(self.linear_layers):            
            if self.lin_layers[index]["output_function"] == "Tanh":
                inputs = F.tanh(layer(inputs))
                #if self.forward_counter % 100 == 0:
                #    print("Linear - Tanh")
            elif self.lin_layers[index]["output_function"] == "Softmax":
                inputs = F.softmax(layer(inputs))
                #if self.forward_counter % 100 == 0:
                #    print("Linear - Softmax")
            elif self.lin_layers[index]["output_function"] == "ReLu":
                inputs = F.relu(layer(inputs))
                #if self.forward_counter % 100 == 0:
                #    print("Linear - ReLu")
                    
        output = self.final_layer(inputs)
        
        #self.forward_counter = self.forward_counter +1
        
        return output #F.softmax(output, dim=1)
