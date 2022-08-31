import torch

a = torch.rand(3,1)
b = torch.rand(3,1)
print(a)
print(b)
c = torch.stack([a,b],dim=0)
print(c)
d = c.reshape(c.shape[0]*c.shape[1],c.shape[2])
print(d)