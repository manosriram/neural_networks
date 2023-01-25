import torch

x = torch.Tensor([3, 4])
y = torch.Tensor([1, 5])

r = torch.rand([2, 5])

flat = r.view([1, 10])

print(flat)
