import torch.nn as nn
import torch.nn.functional as F


class Adjustable_model(nn.Module):
    
    def __init__(self, linear_layers, convolutional_layers = [], output_classes = 10, input_size = (28,28)):
        super(Adjustable_model, self).__init__()
        
        output_size = input_size
        input_size = input_size[0]*input_size[1]
        
        self.convolutional_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        
        self.input_size = input_size
        
        

        if convolutional_layers != []:
            input_size = 1
            for params in convolutional_layers: 
                if "kernel_size" not in params:
                    params['kernel_size'] = 3
                if "stride" not in params:
                    params['stride'] = 1
                if "padding" not in params:
                    params['padding'] = 0
                
                self.convolutional_layers.append(nn.Conv2d(in_channels = input_size, out_channels = params['size'], 
                                                           kernel_size=params['kernel_size'], stride=params['stride'], padding=params['padding']))
                input_size = params['size']
                #output_size = (output_size[0] - (params['kernel_size']-1), output_size[1] - (params['kernel_size']-1))
                output_size = ( int((output_size[0] - (params['kernel_size']) + 2*params['padding'])/params['stride'] +1) , int((output_size[1] - (params['kernel_size']) + 2*params['padding'])/params['stride'] +1))

                
            input_size = input_size*output_size[0]*output_size[1]
        
        
        for size in linear_layers:
            self.linear_layers.append(nn.Linear(input_size, size)) # Linear layers only need layer size as a parameter which is i
            input_size = size
        
        # output layer:
        self.final_layer = nn.Linear(input_size, output_classes)   
    
    def forward(self, inputs):
        
        if len(self.convolutional_layers) != 0:
            for conv_layer in self.convolutional_layers:
                inputs = F.tanh(conv_layer(inputs))
                #nn.MaxPool2d(kernel_size=2, stride=1)
        
        #inputs = torch.flatten(inputs, 1)
        inputs = inputs.view(inputs.size(0), -1)
        for layer in self.linear_layers:
            inputs = F.tanh(layer(inputs))
            

        output = self.final_layer(inputs)

        return output #F.softmax(output, dim=1)

