import torch
import torch.nn as nn
import torch.nn.functional as F

from enum import auto

class Layer:

    class Type:
        CONVOLUTION = auto()
        LINEAR = auto()

    def __init__(self, layer_type: str, output: int, kernal: int=None, stride: int = 1, 
                 padding: int=0, pool: bool=False, batch: bool=False, dropout: float = None):

        self.layer_type = self._get_type(layer_type)
        self.kernals = kernal
        
        if self.layer_type == Layer.Type.CONVOLUTION:
            if self.kernals == None:
                raise TypeError('Kernal size must be set as int for convolution')
            
        self.input = input
        self.output = output
        self.stride = stride
        self.padding = padding
        self.pool = pool
        self.batch = batch
        self.dropout = dropout

        
    
    def _get_type(self, layer_type: str):
        if layer_type == 'conv':
            return Layer.Type.CONVOLUTION
        elif layer_type == 'lin':
            return Layer.Type.LINEAR

class ModelContainer:

    def __init__(self):

        self.container = []
        self.count = 0
    
    def layers(self):
        for idx, x in enumerate(self.container):
            yield idx, x
    
    def add(self, x):
        self.container.append(x)
        self.count += 1
    
    def __len__(self):
        return self.count 
    
    def __str__(self):
        output = ''
        for idx, layer in enumerate(self.container):
            if layer.layer_type == Layer.Type.CONVOLUTION:
                s = 'Convolution'
                c = 'channels'
            else:
                s = 'Linear'
                c = 'Neurons'
            
            output += str('\n')
            output += str('Layer {} is a {} layer, with an output of {} {}: \nPooling is {} \nBatch Normalisation is {} \nDropout is {}'.format(idx+1,s,layer.output,c,layer.pool,layer.batch, layer.dropout))
            output += str('\n')
        return output
    

class ModelMaker(nn.Module):

    def __init__(self, container: ModelContainer, in_features: int, img_size: int):
        
        super(ModelMaker,self).__init__()
        
        self.channels = in_features
        self.curr_size = img_size
        self.network = self._make_layers(container)
        
        

    def _make_layers(self, container: ModelContainer):
        
        layers = []
        
        for idx, layer in container.layers():
            if layer.layer_type == Layer.Type.CONVOLUTION:
                
                layers.append(nn.Conv2d(self.channels,
                                       layer.output,
                                       kernel_size=layer.kernals,
                                       stride=layer.stride,
                                       padding=layer.padding))
                
                if layer.pool:
                    layers.append(nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))
                
                if layer.batch:
                    layers.append(nn.BatchNorm2d(layer.output))
                
                self.curr_size = self._find_input(layer)
                self.channels = layer.output
                
            elif layer.layer_type == Layer.Type.LINEAR:

                if idx == 0:
                    layers.append(nn.Linear(self.channels,
                                            layer.output))
                elif self.channels == 1:
                    layers.append(nn.Linear(self.curr_size,
                                            layer.output))
                else:
                    layers.append(nn.Flatten())
                    layers.append(nn.Linear((self.curr_size**2)*self.channels,
                                            layer.output,
                                            ))
                
                if layer.batch:
                    layers.append(nn.BatchNorm1d(layer.output))

                self.curr_size = layer.output
                self.channels = 1

            if layer.dropout is not None:
                layers.append(nn.Dropout2d(p=layer.dropout))

            if idx < len(container) - 1:
                layers.append(nn.ReLU())

        return nn.Sequential(*layers)

            
    def _find_input(self, layer):
        x = (self.curr_size - layer.kernals + 2*layer.padding)//layer.stride + 1
        if layer.pool:
            return x // 2
        else:
            return x
        
    def forward(self, x):

        x =self.network(x)

        return x

