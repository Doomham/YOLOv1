import torch


def encode(bboxes, labels, S=7, B=2, C=20):
    '''
    :param bboxes: ~(N, 4)
    :param labels: ~(N, )
    :return: ~(S, S, B * 5 + C)
    '''
    xy = (bboxes[:, :2] + bboxes[:, 2:]) * 0.5
    wh = bboxes[:, 2:] - bboxes[:, :2]
    '''
    (448, 448) divides into 7 * 7 grid cells,
    so each of them 64,64
    '''
    cell_size = 448. / S
    grid_cell = xy // cell_size

    xy_norm = (xy - grid_cell * cell_size) / cell_size
    wh_norm = wh / 448.

    target_tensor = torch.zeros(S, S, B * 5 + C)
    '''
    B * 5 + C contains (x, y, w, h, c) * 5 + pr(class|obj) * C
    '''
    for n in range(bboxes.shape[0]):
        i, j = grid_cell[n, :]
        i, j = int(i), int(j)
        for b in range(B):
            target_tensor[i, j, b * 5: b * 5 + 2] = xy_norm[n]
            target_tensor[i, j, b * 5 + 2: b * 5 + 4] = wh_norm[n]
            target_tensor[i, j, b * 5 + 4] = 1.
        target_tensor[i, j, B * 5 + labels[n]] = 1.
    return target_tensor
