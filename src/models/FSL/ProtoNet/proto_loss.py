import torch
from torch.nn import functional as F
from torch.nn.modules import Module

from config.consts import General as _CG


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, x, target):
        return prototypical_loss(x, target, self.n_support)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(recons, target, n_support):
    '''
    Inspired by https://github.com/jakesnell/prototypical-networks/blob/master/protonets/models/few_shot.py
    Compute the barycentres by averaging the features of n_support
    samples for each class in target, computes then the distances from each
    samples' features to each one of the barycentres, computes the
    log_probability for each n_query samples for each one of the current
    classes, of appartaining to a class c, loss and accuracy are then computed
    and returned
    Args:
    - recons: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    - n_support: number of samples to keep in account when computing
      barycentres, for each one of the current classes
    '''
    target_cpu = target.to('cpu')
    input_cpu = recons.to('cpu')

    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    n_classes = len(classes)
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = recons.to('cpu')[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

    return loss_val, acc_val


def proto_test(recons, target, n_support):
    classes = torch.unique(target)
    n_classes = len(classes)
    mapping = {i: classes[i].item() for i in range(n_classes)}

    # assuming n_query, n_target constants
    numel_set = len(torch.nonzero(target == classes[0]).view(-1))   # numel for support + query
    n_query = numel_set - n_support

    # retrieve support and query indexes
    support_idxs, query_idxs = [], torch.LongTensor().to(_CG.DEVICE)
    for c in classes:
        s, q = torch.split(torch.nonzero(target == c).view(-1), [n_support, n_query])
        support_idxs.append(s)  # 3 tensors with 5 samples each
        query_idxs = torch.cat((query_idxs, q))

    # use retrieved indexes to compute mean of 5 (idx_list) elements per class (output.size = n_classes * flatten_features)
    prototypes = torch.stack([recons[idx_list].mean(0) for idx_list in support_idxs])
    query_samples = recons[query_idxs.view(-1)]
    dists = euclidean_dist(query_samples, prototypes)   # dim: (n_cls * sam_per_class, n_classes)

    # softmax of negative distance otherwise the softmax is negative (the highest value must be the closest)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes).to(_CG.DEVICE)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    _, y_hat = log_p_y.max(2)
    acc_overall = y_hat.eq(target_inds.squeeze(2)).float().mean()
    acc_vals = { c: y_hat[c].eq(target_inds.squeeze(2)[c]).float().mean() for c in range(n_classes) }

    return acc_overall, { v: acc_vals[i] for i, v in enumerate(mapping.values()) }
