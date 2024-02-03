# This is used for the advanced model creator to add or remove layers

import matplotlib.pyplot as plt
import matplotlib

class layer_box_representation():
    def __init__(self):
        self.conv_layers = []
        self.lin_layers = []

    def input_box(self, axes, x, y):
        left, width = x, .3
        bottom, height = y, .15
        right = left + width
        top = bottom + height
        
        p = plt.Rectangle((left, bottom), width, height, fill=True, edgecolor = 'red', facecolor='lightcoral')
        p.set_transform(axes.transAxes)
        p.set_clip_on(False)
        axes.add_patch(p)
        
        axes.text(0.5 * (left + right), 0.5 * (bottom + top), "Input Layer",
            horizontalalignment='center',
            verticalalignment='center',
            #bbox=props,
            transform=axes.transAxes)

    def output_box(self, axes, x, y):
        left, width = x, .3
        bottom, height = y, .15
        right = left + width
        top = bottom + height
        
        p = plt.Rectangle((left, bottom), width, height, fill=True, edgecolor = 'green', facecolor='lightgreen')
        p.set_transform(axes.transAxes)
        p.set_clip_on(False)
        axes.add_patch(p)
        
        axes.text(0.5 * (left + right), 0.5 * (bottom + top), "Output Layer",
            horizontalalignment='center',
            verticalalignment='center',
            #bbox=props,
            transform=axes.transAxes)
        
    def conv_box(self, axes, x, y, text_str):
        left, width = x, .54
        bottom, height = y, .38
        right = left + width
        top = bottom + height
        
        p = plt.Rectangle((left, bottom), width, height, fill=True, edgecolor = 'yellow', facecolor='lightyellow')
        p.set_transform(axes.transAxes)
        p.set_clip_on(False)
        axes.add_patch(p)
    
        #props = dict(boxstyle='round', facecolor='grey', alpha=0.25)
        axes.text(0.5 * (left + right), 0.5 * (bottom + top), text_str,
            horizontalalignment='center',
            verticalalignment='center',
            #bbox=props,
            transform=axes.transAxes)

    def lin_box(self, axes, x, y, text_str):
        left, width = x, .54
        bottom, height = y, .25
        right = left + width
        top = bottom + height
        
        p = plt.Rectangle((left, bottom), width, height, fill=True, edgecolor = '#1f77b4', facecolor='aliceblue')
        p.set_transform(axes.transAxes)
        p.set_clip_on(False)
        axes.add_patch(p)
    
        #props = dict(boxstyle='round', facecolor='grey', alpha=0.25)
        axes.text(0.5 * (left + right), 0.5 * (bottom + top), text_str,
            horizontalalignment='center',
            verticalalignment='center',
            #bbox=props,
            transform=axes.transAxes)

    def conv_string_concat(self, layer_number, layer_dict):
        convolutional_cells, kernel_size, padding, stride, output_function, pooling = layer_dict["size"], layer_dict["kernel_size"], layer_dict["padding"], layer_dict["stride"], layer_dict["output_function"], layer_dict["pooling"]
        
        layer_number_str =        f"Convolutional Layer {layer_number+1}:  \n \n"
        convolutional_cells_str = f"Convolutional Cells:  {convolutional_cells} \n"
        kernel_size_str =         f"Kernel Size:               {kernel_size} \n"
        padding_str =             f"Padding:                    {padding} \n"
        stride_str =              f"Stride:                       {stride} \n"
        output_function_str =     f"Output Function:       {output_function} \n"
        pooling_str =             f"Pooling:                     {pooling} "
    
        return layer_number_str + convolutional_cells_str + kernel_size_str + padding_str + stride_str + output_function_str + pooling_str

    def lin_string_concat(self, layer_number, layer_dict):
        output_function = layer_dict["output_function"]
        lin_cells = layer_dict["linear_cells"]
        
        layer_number_str =        f"Linear Layer {layer_number+1}:  \n \n"
        lin_cells_str =           f"Linear Cells:  {lin_cells} \n"
        output_function_str =     f"Output Function: {output_function} \n"
    
        return layer_number_str + lin_cells_str + output_function_str
    
    def add_arrow(self, axes, x, y):
        p = plt.Arrow(x, y, dy=0, dx=0.05, width=0.05, fill=False, edgecolor = 'black')
        p.set_transform(axes.transAxes)
        p.set_clip_on(False)
        axes.add_patch(p)    
    
    def display_current_boxes(self):
        conv_layers = self.conv_layers
        lin_layers = self.lin_layers
        
        fig, ax = plt.subplots()#figsize=(8,3))
        fig.tight_layout()
    
        plt.axis('scaled')
        plt.axis('off')

        # add input box:
        self.input_box(ax, 0.0, 0.82)
    
        for index, layer in enumerate(conv_layers):
            textstr = self.conv_string_concat(layer_number=index, layer_dict=layer)
            x_arrow = 0.3+index*0.6
            x_box = x_arrow+0.05
            self.add_arrow(ax, x_arrow, 0.9)
            self.conv_box(ax, x_box, 0.7, textstr)
        self.add_arrow(ax, 0.3+len(conv_layers)*0.6, 0.9)

        for index, layer in enumerate(lin_layers):
            textstr = self.lin_string_concat(layer_number=index, layer_dict=layer)
            x_arrow = 0.3+index*0.6 + len(conv_layers)*0.6
            x_box = x_arrow+0.05
            self.add_arrow(ax, x_arrow, 0.9)
            self.lin_box(ax, x_box, 0.75, textstr)

        self.add_arrow(ax, 0.3+len(lin_layers)*0.6+len(conv_layers)*0.6, 0.9)
        self.output_box(ax, 0.35+len(lin_layers)*0.6+len(conv_layers)*0.6, 0.82)  

        #plt.savefig('layerboxes.png')
        return fig
        #plt.show()

    def add_conv_layer(self, conv_layer):
        self.conv_layers.append(conv_layer)

    def remove_conv_layer(self):
        self.conv_layers.pop()

    def add_lin_layer(self, lin_layers):
        self.lin_layers.append(lin_layers)

    def remove_lin_layer(self):
        self.lin_layers.pop()
        
    def get_conv_layers(self):
        return self.conv_layers

    def get_lin_layers(self):
        return self.lin_layers