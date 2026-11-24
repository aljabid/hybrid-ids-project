import torch
import torch.nn as nn
class Autoencoder(nn.Module):
    def __init__(self, input_dim=10, hidden_dims=None):
        super(Autoencoder, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]
        enc_layers = []
        in_dim = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.encoder = nn.Sequential(*enc_layers)
        dec_dims   = list(reversed(hidden_dims[:-1])) + [input_dim]
        dec_layers = []
        in_dim     = hidden_dims[-1]
        for i, h in enumerate(dec_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            if i < len(dec_dims) - 1:
                dec_layers.append(nn.ReLU())
            else:
                dec_layers.append(nn.Sigmoid())
            in_dim = h
        self.decoder = nn.Sequential(*dec_layers)
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    def encode(self, x):
        return self.encoder(x)
    def reconstruction_error(self, x):
        with torch.no_grad():
            x_hat = self.forward(x)
            error = torch.mean((x - x_hat) ** 2, dim=1)
        return error
