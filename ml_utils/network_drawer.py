from matplotlib import pyplot
from math import cos, sin, atan


class Neuron():
    def __init__(self, x, y, layerType="Linear Layer"):
        self.x = x
        self.y = y
        self.layerType = layerType

    def draw(self, neuron_radius):
        if self.layerType == "Linear Layer":
            color = '#1f77b4'
        elif self.layerType == "Convolutional Layer":
            color = 'yellow'
        elif self.layerType == "Input Layer":
            color = 'red'
        elif self.layerType == "Output Layer":
            color = 'green'
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=True, color=color)
        pyplot.gca().add_patch(circle)

class Layer():
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer, layerType):
        self.vertical_distance_between_layers = 2
        self.horizontal_distance_between_layers = 8
        self.neuron_radius = 0.5
        self.layer_type = layerType
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.x = self.__calculate_layer_x_position()
        self.neurons = self.__intialise_neurons(number_of_neurons, self.layer_type)

    def __intialise_neurons(self, number_of_neurons, layerType):
        neurons = []
        y = self.__calculate_top_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(self.x, y, layerType)
            neurons.append(neuron)
            y += self.vertical_distance_between_layers
        return neurons

    def __calculate_top_margin_so_layer_is_centered(self, number_of_neurons):
        return self.vertical_distance_between_layers * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_x_position(self):
        if self.previous_layer:
            return self.previous_layer.x + self.horizontal_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2):
        angle = atan((neuron2.y - neuron1.y) / float(neuron2.x - neuron1.x))
        x_adjustment = self.neuron_radius * cos(angle)
        y_adjustment = self.neuron_radius * sin(angle)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment), (neuron1.y - y_adjustment, neuron2.y + y_adjustment), linewidth=0.1)
        pyplot.gca().add_line(line)

    def draw(self, layerNumber=0):
        for neuron in self.neurons:
            neuron.draw( self.neuron_radius )
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(neuron, previous_layer_neuron)
        # write Text
        y_text = self.number_of_neurons_in_widest_layer * self.vertical_distance_between_layers
        if self.layer_type == 'Input Layer':
            pyplot.text(self.x, y_text, 'Input\nLayer', fontsize = 10, 
                        horizontalalignment='center',
                        verticalalignment='center')
        elif self.layer_type == 'Output Layer':
            pyplot.text(self.x, y_text, 'Output\nLayer', fontsize = 10, 
                        horizontalalignment='center',
                        verticalalignment='center')
        elif self.layer_type == 'Convolutional Layer':
            pyplot.text(self.x, y_text, 'Convolutional\nLayer'+str(layerNumber), fontsize = 8, 
                        horizontalalignment='center',
                        verticalalignment='center')
        else:
            pyplot.text(self.x, y_text, 'Linear\nLayer '+str(layerNumber), fontsize = 10, 
                        horizontalalignment='center',
                        verticalalignment='center')

class NeuralNetwork():
    def __init__(self, number_of_neurons_in_widest_layer):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        #self.conv_count = 1 # not needed
        self.lin_count = 1

    def add_layer(self, number_of_neurons, layerType="Linear Layer"):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer, layerType )
        self.layers.append(layer)

    def draw(self):
        fig = pyplot.figure(figsize=(6, 20), dpi=80)
        for i in range( len(self.layers) ):
            layer = self.layers[i]
            if layer.layer_type == "Linear Layer":
                i = self.lin_count
                self.lin_count = self.lin_count + 1
            layer.draw(i)
        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title('Your Neural Network', fontsize=15 )
        #pyplot.show()
        return fig
        

class DrawNN():
    def __init__( self, neural_network, conv_layers ):
        self.neural_network = neural_network
        self.conv_layers = conv_layers

    def draw( self ):
        widest_layer = max( self.neural_network )
        network = NeuralNetwork( widest_layer )
        for ind, l in enumerate(self.neural_network):
            if ind == 0:
                layertype = "Input Layer"
                network.add_layer(l, layerType=layertype)
            elif ind == len(self.neural_network)-1:
                layertype = "Output Layer"
                network.add_layer(l, layerType=layertype)
            elif ind >= 1 and ind <= self.conv_layers and ind < len(self.neural_network)-1:
                layertype = "Convolutional Layer"
                network.add_layer(l, layerType=layertype)
            elif ind >= 1 and ind > self.conv_layers and ind < len(self.neural_network)-1:
                layertype = "Linear Layer"
                network.add_layer(l, layerType=layertype)
        return network.draw()