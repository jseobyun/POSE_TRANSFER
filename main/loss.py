import torch

def L1_loss(pred, target):
    loss = torch.abs(pred-target)
    loss = torch.sum(loss, dim=(1, 2, 3))
    loss = torch.mean(loss)

    return loss

def BCE_loss(pred, loss_type):
    B, C = pred.shape
    BCE = torch.nn.BCELoss()
    if loss_type =='fake':
        label = torch.zeros([B,1]).cuda()
    elif loss_type =='real':
        label = torch.ones([B,1]).cuda()
    loss =BCE(pred, label)
    return loss
