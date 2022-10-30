import torch.nn as nn
from torchmeta.modules import MetaLinear, MetaModule

def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

def conv_3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class PrototypicalNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=64):
        super(PrototypicalNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, out_channels)
        )

    def forward(self, inputs, params = None):
        #print(inputs.shape)
        embeddings = self.encoder(inputs.view(-1, *inputs.shape[2:]))
        return embeddings.view(*inputs.shape[:2], -1)


class PrototypicalNetwork_new(nn.Module):
    def __init__(self, in_channels, out_channels, num_ways, hidden_size=64):
        super(PrototypicalNetwork_new, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size

        self.encoder = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, out_channels)
        )
        self.extended_encoder = nn.Sequential(
            conv_3x3(out_channels, 25)
        )

        self.classifier = nn.Linear(25, num_ways)

    def forward_old(self, inputs, params=None):
        embeddings = self.encoder(inputs.view(-1, *inputs.shape[1:]))
        logits = self.classifier(embeddings.view(*embeddings.shape[:1],-1))#, params=self.get_subdict(params, 'classifier'))
        return logits.view(*inputs.shape[:1], -1)
        
    def forward_enc(self, inputs, params = None) :
        embeddings = self.encoder(inputs.view(-1, *inputs.shape[2:]))
        return embeddings.view(*inputs.shape[:2], -1)

    def forward(self, inputs, params = None) :
        embeddings = self.encoder(inputs.view(-1, *inputs.shape[1:]))
        embeddings = self.extended_encoder(embeddings)
        logits = self.classifier(embeddings.view(*embeddings.shape[:1],-1))
        return logits.view(*inputs.shape[:1], -1)
