import torch
import numpy as np


def compute_iou(bbox1, bbox2):
    '''
    :param bbox1: ~(m, 4)
    :param bbox2: ~(n,4)
    :return: ~(m, n)
    '''
    m = bbox1.shape[0]
    n = bbox2.shape[0]

    tl = torch.min(bbox1[:, :2].unsqueeze(1).expand(m, n, 2),
                   bbox2[:, :2].unsqueeze(0).expand(m, n, 2))
    br = torch.max(bbox1[:, 2:].unsqueeze(1).expand(m, n, 2),
                   bbox1[:, 2:].unsqueeze(0).expand(m, n, 2))
    overlaps = torch.prod(br - tl, dim=2)

    area1 = torch.prod(bbox1[:, 2:] - bbox1[:, :2], dim=1)
    area2 = torch.prod(bbox2[:, 2:] - bbox2[:, :2], dim=1)

    return overlaps / (-overlaps + area2 + area1.unsqueeze(0))


arr = torch.arange(12).view(3, 4)
print(arr.unsqueeze(1).shape)