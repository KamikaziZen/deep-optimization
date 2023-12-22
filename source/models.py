import torch.nn as nn
import torch
import math
import itertools


class Class_ReLU(nn.Module):
    def __init__(self, bias=False):
        
        super().__init__()
        
        self.linear1 = nn.Linear(in_features=256, out_features=300, bias=bias)
        self.linear2 = nn.Linear(in_features=300, out_features=300, bias=bias)
        self.linear3 = nn.Linear(in_features=300, out_features=10, bias=bias)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        return self.linear3(x)
    

class AE_Sigm(nn.Module):
    def __init__(self, bias=False):
        super().__init__()
         
        self.encoder = torch.nn.Sequential(
            nn.Linear(in_features=256, out_features=300, bias=bias),
            nn.Sigmoid(),
            nn.Linear(in_features=300, out_features=100, bias=bias),
            nn.Sigmoid(),
            nn.Linear(in_features=100, out_features=20, bias=bias),
            nn.Sigmoid(),
        )
         
        self.decoder = torch.nn.Sequential(
            nn.Linear(in_features=20, out_features=100, bias=bias),
            nn.Sigmoid(),
            nn.Linear(in_features=100, out_features=300, bias=bias),
            nn.Sigmoid(),
            nn.Linear(in_features=300, out_features=256, bias=bias),
        )
        
        for m in itertools.chain(self.encoder.modules(), self.decoder.modules()):
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, 
                                 a=-1/math.sqrt(m.in_features), 
                                 b=1/math.sqrt(m.in_features))
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AE_ReLU(nn.Module):
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
            # this shouldn't be here, right? 
            # nn.ReLU(),
        )
        
        for m in itertools.chain(self.encoder.modules(), self.decoder.modules()):
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, 
                                 a=-1/math.sqrt(m.in_features), 
                                 b=1/math.sqrt(m.in_features))
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    
class AE_ReLU_Small(nn.Module):
    def __init__(self, bias=False, zeros=False, uniform=False):
        super().__init__()
         
        self.encoder = torch.nn.Sequential(
            nn.Linear(in_features=256, out_features=100, bias=bias),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=20, bias=bias),
            nn.ReLU(),
        )
         
        self.decoder = torch.nn.Sequential(
            nn.Linear(in_features=20, out_features=100, bias=bias),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=256, bias=bias),
            # this shouldn't be here, right? 
            # nn.ReLU(),
        )
        
        if zeros:
            for m in itertools.chain(self.encoder.modules(), self.decoder.modules()):
                if isinstance(m, nn.Linear):
                    nn.init.zeros_(m.weight)
        if uniform:
            for m in itertools.chain(self.encoder.modules(), self.decoder.modules()):
                if isinstance(m, nn.Linear):
                    nn.init.uniform_(m.weight)
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    
class AE_Tanh(nn.Module):
    def __init__(self, bias=False):
        super().__init__()
         
        self.encoder = torch.nn.Sequential(
            nn.Linear(in_features=256, out_features=300, bias=bias),
            nn.Tanh(),
            nn.Linear(in_features=300, out_features=100, bias=bias),
            nn.Tanh(),
            nn.Linear(in_features=100, out_features=20, bias=bias),
            nn.Tanh(),
        )
         
        self.decoder = torch.nn.Sequential(
            nn.Linear(in_features=20, out_features=100, bias=bias),
            nn.Tanh(),
            nn.Linear(in_features=100, out_features=300, bias=bias),
            nn.Tanh(),
            nn.Linear(in_features=300, out_features=256, bias=bias),
            # this shouldn't be here, right? 
            # nn.Tanh(),
        )
        
        for m in itertools.chain(self.encoder.modules(), self.decoder.modules()):
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, 
                                 a=-1/math.sqrt(m.in_features), 
                                 b=1/math.sqrt(m.in_features))
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    
