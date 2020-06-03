import torch
import torch.utils.data as data
from data.vocdataset import VOCDetection
from torch import optim
from yolo import YOLOv1
from loss import compute_loss
import math
from backbone.darknet import darknet19
from utils.augmentatio import SSDAugmentation, detection_collate
import time
from utils.target_creator import encode

print_freq = 5
init_lr = 0.001
base_lr = 0.01
momentum = 0.9
weight_decay = 5.0e-4
num_epochs = 135
batch_size = 64


def update_lr(optimizer, epoch, burnin_base, burnin_exp=4.0):
    if epoch == 0:
        lr = init_lr + (base_lr - init_lr) * math.pow(burnin_base, burnin_exp)
    elif epoch == 1:
        lr = base_lr
    elif epoch == 75:
        lr = 0.001
    elif epoch == 105:
        lr = 0.0001
    else:
        return

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def save(model):
    save_dict = dict()
    save_dict['model'] = model.state_dict()
    timestr = time.strftime('%m%d%H%M')
    save_path = 'yolo_%s' % timestr
    torch.save(save_dict, save_path)
    return save_path



def train():
    dark_net = darknet19(True).cuda()
    yolo = YOLOv1(dark_net)
    yolo.cuda()

    optimizer = optim.SGD(yolo.parameters(), lr=init_lr, momentum=momentum, weight_decay=weight_decay)

    #load dataset
    VOC_ROOT = "D:/pyworks/FasterRCNN/data/VOCdevkit/"
    data_set = VOCDetection(VOC_ROOT, transform=SSDAugmentation([448, 448], mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)))
    data_loader = data.DataLoader(data_set, batch_size=2, shuffle=True, collate_fn=detection_collate, pin_memory=True)

    #start_time = time.time()

    for epoch in range(num_epochs):
        loss_per_batch = 0
        for i, (images, targets) in enumerate(data_loader):
            update_lr(optimizer, epoch, float(i) / float(len(data_loader) - 1))
            lr = get_lr(optimizer)

            predicted_tensor = yolo(images)
            target_tenor = []
            for k in range(images.shape[0]):
                bboxes = targets[:, :4]
                labels = targets[:, 4]
                target_tenor.append(encode(bboxes, labels))

            loss = compute_loss(predicted_tensor, target_tenor)
            loss_per_batch += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % print_freq == 0 :
            print('Epoch [%d/%d], Loss: %.4f'
                  % (epoch, num_epochs, loss_per_batch / batch_size))

        if epoch % 10 and epoch >= 10:
            save(yolo)

train()