import torch.nn as nn
import torch.nn.functional as F

class WeightStandarization(nn.Module):
    def forward(self, weight):
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return weight
         
    
class WeightStandarization1d(nn.Module):
    def forward(self, weight):
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return weight
    
    
class WeightStandarization0d(nn.Module):
    def forward(self, weight):
        weight_mean = weight.mean(dim=1, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return weight