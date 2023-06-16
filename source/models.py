import torch.nn as nn
import torch
import math
import itertools


# class AE(nn.Module):
#     def __init__(self):
#         super().__init__()
         
#         self.encoder = torch.nn.Sequential(
#             nn.Linear(in_features=256, out_features=300),
#             nn.Sigmoid(),
#             nn.Linear(in_features=300, out_features=100),
#             nn.Sigmoid(),
#             nn.Linear(in_features=100, out_features=20),
#             nn.Sigmoid(),
#         )
         
#         self.decoder = torch.nn.Sequential(
#             nn.Linear(in_features=20, out_features=100),
#             nn.Sigmoid(),
#             nn.Linear(in_features=100, out_features=300),
#             nn.Sigmoid(),
#             nn.Linear(in_features=300, out_features=256),
#         )
        
#         for m in itertools.chain(self.encoder.modules(), self.decoder.modules()):
#             if isinstance(m, nn.Linear):
#                 nn.init.uniform_(m.weight, 
#                                  a=-1/math.sqrt(m.in_features), 
#                                  b=1/math.sqrt(m.in_features))
 
#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded


class AE(nn.Module):
    def __init__(self, bias=False):
        super().__init__()
         
        self.encoder = torch.nn.Sequential(
            nn.Linear(in_features=256, out_features=300, bias=bias),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=100, bias=bias),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=20, bias=bias),
            nn.ReLU(),
        )
         
        self.decoder = torch.nn.Sequential(
            nn.Linear(in_features=20, out_features=100, bias=bias),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=300, bias=bias),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=256, bias=bias),
            nn.ReLU(),
        )
        
        # for m in itertools.chain(self.encoder.modules(), self.decoder.modules()):
        #     if isinstance(m, nn.Linear):
        #         nn.init.uniform_(m.weight, 
        #                          a=-1/math.sqrt(m.in_features), 
        #                          b=1/math.sqrt(m.in_features))
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded