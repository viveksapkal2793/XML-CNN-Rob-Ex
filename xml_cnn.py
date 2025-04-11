import torch
from torch import nn as nn
from torch.nn import functional as F

from my_functions import out_size
from adversarial_defense import FeatureSqueezing

class xml_cnn(nn.Module):
    def __init__(self, params, embedding_weights):
        super(xml_cnn, self).__init__()

        self.params = params

        stride = params["stride"]
        emb_dim = embedding_weights.shape[1]
        hidden_dims = params["hidden_dims"]
        sequence_length = params["sequence_length"]
        filter_channels = params["filter_channels"]
        d_max_pool_p = params["d_max_pool_p"]
        self.filter_sizes = params["filter_sizes"]

        # Initialize feature squeezing for inference
        self.feature_squeezing = FeatureSqueezing(bit_depth=params.get("bit_depth", 8))

        self.lookup = nn.Embedding.from_pretrained(embedding_weights, freeze=False)

        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.fin_l_out_size = 0

        self.dropout_0 = nn.Dropout(0.5)
        self.dropout_1 = nn.Dropout(0.5)

        for fsz, n, ssz in zip(self.filter_sizes, d_max_pool_p, stride):
            conv_n = nn.Conv2d(
                1, filter_channels, (fsz, emb_dim), stride=(ssz, emb_dim)
            )
            # Initialize with He's method
            torch.nn.init.kaiming_normal_(conv_n.weight)

            # Dynamic Max-Pooling
            conv_out_size = out_size(sequence_length, fsz, filter_channels, stride=ssz)
            assert conv_out_size % n == 0
            pool_k_size = conv_out_size // n
            pool_n = nn.MaxPool1d(pool_k_size, stride=pool_k_size)

            self.fin_l_out_size += n

            self.conv_layers.append(conv_n)
            self.pool_layers.append(pool_n)
        
        # Create linear layers after calculating correct dimensions
        self.l1 = nn.Linear(self.fin_l_out_size, hidden_dims)
        self.l2 = nn.Linear(hidden_dims, params["num_of_class"])

        # Initialize with He's method
        torch.nn.init.kaiming_normal_(self.l1.weight)
        torch.nn.init.kaiming_normal_(self.l2.weight)

    def forward(self, x, apply_squeezing=False):
        # Embedding layer
        h_non_static = self.lookup.forward(x)  # Remove permute to handle batch correctly
        h_non_static = h_non_static.unsqueeze(1)  # Add channel dimension (batch, 1, seq_len, emb_dim)
        h_non_static = self.dropout_0(h_non_static)

        h_list = []

        # Conv, Pooling layers
        for i in range(len(self.filter_sizes)):
            h_n = self.conv_layers[i](h_non_static)
            h_n = h_n.view(h_n.shape[0], 1, h_n.shape[1] * h_n.shape[2])
            h_n = self.pool_layers[i](h_n)
            h_n = F.relu(h_n)
            h_n = h_n.view(h_n.shape[0], -1)
            h_list.append(h_n)

        if len(self.filter_sizes) > 1:
            h = torch.cat(h_list, 1)
        else:
            h = h_list[0]
        
        # Ensure dimensions match what's expected
        assert h.shape[1] == self.fin_l_out_size, f"Expected features: {self.fin_l_out_size}, got: {h.shape[1]}"
        
        # Full connected layer
        h = F.relu(self.l1(h))
        h = self.dropout_1(h)
        
        # Output layer
        y = self.l2(h)
        # Apply feature squeezing at inference time if requested
        if apply_squeezing:
            y = self.feature_squeezing.squeeze(y)

        return y
    
    def predict(self, x):
        """Inference method with feature squeezing defense"""
        self.eval()
        with torch.no_grad():
            return self.forward(x, apply_squeezing=True)