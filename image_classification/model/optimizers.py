import torch.optim as optim

def get_optimizer(net, lr=0.001, momentum=0.9):
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    return optimizer
