import torch.nn as nn

class attention_manager(object):
    def __init__(self, model, multi_gpu):        
        self.multi_gpu = multi_gpu
        
        self.attention = []
        self.handler = []

        self.model = model
        
        if multi_gpu:
            self.register_hook(self.module.model)
        else:
            self.register_hook(self.model)
            

    def register_hook(self, model):
        def get_attention_features(_, inputs, outputs):
            self.attention.append(outputs)
        
        for name, layer in model._modules.items():
            # but recursively register hook on all it's module children
            if isinstance(layer, nn.Sequential):
                self.register_hook(layer)
            else:
                for name, layer2 in layer._modules.items():
                    if name == 'attention':
                        handle = layer2.register_forward_hook(get_attention_features)
                        self.handler.append(handle)
          
    
    def remove_hook(self):
        for handler in self.handler:
            handler.remove()
        