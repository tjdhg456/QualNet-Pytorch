import torch.nn as nn

class Wrapper(nn.Module):
    def __init__(self, model, decoder):
        super(Wrapper, self).__init__()
        self.model = model
        self.decoder = decoder
        
    def forward(self, x, train=True):
        if train:
            out, out_bij = self.model(x, train=True)
            HR_img_gen = self.decoder.inverse(out_bij)
            return out, HR_img_gen
        else:
            out = self.model(x, train=False)
            return out
