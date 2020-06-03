import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.bboxtools import compute_iou
from torch.autograd import Variable


def compute_loss(pred_tensor, target_tensor,
                 S=7, B=2, C=20, lambda_noobj=0.5, lambda_coor=5):
    '''
    :param pred_tensor: ~(N, S, S, 5 * B + C)
    :param target_tensor: ~(N, S, S, 5 * B + C)
    B * 5 + C ~ (x, y, w, h, c) + pr(class|obj)
    :return: ~(1, )
    '''
    N = pred_tensor.shape[0]
    obj_mask = (target_tensor[:, :, :, 4] > 0)  #~(N, S, S)
    noobj_mask = (target_tensor[:, :, :, 4] == 0)
    obj_mask = obj_mask.unsqueeze(-1).expand(N, S, S, B * 5 + C)
    noobj_mask = noobj_mask.unsqueeze(-1).expand(N, S, S, B * 5 + C)

    obj_pred = pred_tensor[obj_mask].view(-1, B * 5 + C)
    noobj_pred = pred_tensor[noobj_mask].view(-1, B * 5 + C)
    obj_target = target_tensor[obj_mask].view(-1, B * 5 + C)
    noobj_target = target_tensor[noobj_mask].view(-1, B * 5 + C)

    '''compute noobj loss'''
    noobj_pred_c = noobj_pred[:, 4: B * 5: 5]
    noobj_target_c = noobj_target[:, 4: B * 5: 5]
    noobj_loss = torch.sum((noobj_target_c - noobj_pred_c)**2)

    '''compute obj loss'''
    bbox_pred = obj_pred[:, : 5 * B].contiguous().view(-1, 5)
    class_pred = obj_pred[:, 5 * B:]
    bbox_target = obj_target[:, : 5 * B].contiguous().view(-1, 5)
    class_target = obj_target[:, 5 * B:]

    # Compute loss for the cells with objects.
    coord_response_mask = torch.cuda.ByteTensor(bbox_target.size()).fill_(0)  # [n_coord x B, 5]
    coord_not_response_mask = torch.cuda.ByteTensor(bbox_target.size()).fill_(1)  # [n_coord x B, 5]
    bbox_target_iou = torch.zeros(bbox_target.size()).cuda()  # [n_coord x B, 5], only the last 1=(conf,) is used

    # Choose the predicted bbox having the highest IoU for each target bbox.
    for i in range(0, bbox_target.size(0), B):
        pred = bbox_pred[i:i + B]  # predicted bboxes at i-th cell, [B, 5=len([x, y, w, h, conf])]
        pred_xyxy = Variable(torch.FloatTensor(pred.size()))  # [B, 5=len([x1, y1, x2, y2, conf])]
        # Because (center_x,center_y)=pred[:, 2] and (w,h)=pred[:,2:4] are normalized for cell-size and image-size respectively,
        # rescale (center_x,center_y) for the image-size to compute IoU correctly.
        pred_xyxy[:, :2] = pred[:, 2] / float(S) - 0.5 * pred[:, 2:4]
        pred_xyxy[:, 2:4] = pred[:, 2] / float(S) + 0.5 * pred[:, 2:4]

        target = bbox_target[
            i]  # target bbox at i-th cell. Because target boxes contained by each cell are identical in current implementation, enough to extract the first one.
        target = bbox_target[i].view(-1, 5)  # target bbox at i-th cell, [1, 5=len([x, y, w, h, conf])]
        target_xyxy = Variable(torch.FloatTensor(target.size()))  # [1, 5=len([x1, y1, x2, y2, conf])]
        # Because (center_x,center_y)=target[:, 2] and (w,h)=target[:,2:4] are normalized for cell-size and image-size respectively,
        # rescale (center_x,center_y) for the image-size to compute IoU correctly.
        target_xyxy[:, :2] = target[:, 2] / float(S) - 0.5 * target[:, 2:4]
        target_xyxy[:, 2:4] = target[:, 2] / float(S) + 0.5 * target[:, 2:4]

        iou = compute_iou(pred_xyxy[:, :4], target_xyxy[:, :4])  # [B, 1]
        max_iou, max_index = iou.max(0)
        max_index = max_index.data.cuda()

        coord_response_mask[i + max_index] = 1
        coord_not_response_mask[i + max_index] = 0

        # "we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth"
        # from the original paper of YOLO.
        bbox_target_iou[i + max_index, torch.LongTensor([4]).cuda()] = (max_iou).data.cuda()
    bbox_target_iou = Variable(bbox_target_iou).cuda()

    # BBox location/size and objectness loss for the response bboxes.
    bbox_pred_response = bbox_pred[coord_response_mask].view(-1, 5)  # [n_response, 5]
    bbox_target_response = bbox_target[coord_response_mask].view(-1,
                                                                 5)  # [n_response, 5], only the first 4=(x, y, w, h) are used
    target_iou = bbox_target_iou[coord_response_mask].view(-1, 5)  # [n_response, 5], only the last 1=(conf,) is used
    loss_xy = F.mse_loss(bbox_pred_response[:, :2], bbox_target_response[:, :2], reduction='sum')
    loss_wh = F.mse_loss(torch.sqrt(bbox_pred_response[:, 2:4]), torch.sqrt(bbox_target_response[:, 2:4]),
                         reduction='sum')
    loss_obj = F.mse_loss(bbox_pred_response[:, 4], target_iou[:, 4], reduction='sum')

    # Class probability loss for the cells which contain objects.
    loss_class = F.mse_loss(class_pred, class_target, reduction='sum')

    # Total loss
    loss = lambda_coor * (loss_xy + loss_wh) + loss_obj + lambda_noobj * noobj_loss + loss_class
    loss = loss / float(N)
'''
    loss1 = torch.sum((bbox_pred[:, 0: 5 * B: 5] - bbox_target[:, 0: 5 * B: 5]) ** 2
                      + (bbox_pred[:, 1: 5 * B: 5] - bbox_target[:, 1: 5 * B: 5]) ** 2)
    loss2 = torch.sum((torch.sqrt(bbox_pred[:, 2: 5 * B: 5].float())
                       - torch.sqrt(bbox_target[:, 2: 5 * B: 5].float())) ** 2
                      + (torch.sqrt(bbox_pred[:, 3: 5 * B: 5].float())
                         - torch.sqrt((bbox_target[:, 3: 5 * B: 5].float()))) ** 2)
    loss3 = torch.sum((bbox_pred[:, 4: B * 5: 5] - bbox_target[:, 4 : B * 5: 5]) ** 2)
    loss4 = torch.sum(((class_target * class_pred) - class_target) ** 2)

    return (lambda_coor*(loss1 + loss2) + loss3 + loss4 + lambda_noobj * noobj_loss) / N'''


'''
arr1 = torch.rand((3, 2))
arr2 = torch.rand((3, 2))
import torch.nn.functional as F
print(F.mse_loss(arr1, arr2, reduction='sum'))
print(torch.sum((arr1 - arr2)**2))
'''
'''
arr = torch.arange(12).view(2, 6)
arr[:, 1:7:2] = 1
print(arr)'''
'''
brr = torch.arange(12)
print(brr)
brr[1:9:4] = -1
print(brr)
'''
'''
arr1 = torch.rand((5, 7, 7, 30))
arr2 = torch.rand((5, 7, 7, 30))
a1, a2 = compute_loss(arr1, arr2)
print(a1.shape)
print(a2.shape)
'''
'''
arr = torch.arange(24).view(1, 2, 3, 4)
arr -= 12
mask = (arr[:, :, :, 3] > 0)
new_mask = mask.unsqueeze(-1).expand_as(arr)
print(mask)
print(arr)
print(arr[:, :, :, 3])
print(arr[new_mask])
print(arr[mask])
'''
'''
arr = torch.tensor([[1, -1, 0, 3], [1, -1, 0, -3]])
mask = arr[:, 3]>0
print(arr[mask.unsqueeze(-1).expand_as(arr)])'''