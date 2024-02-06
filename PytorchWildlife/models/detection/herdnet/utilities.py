import numpy as np
import torch
import torchvision
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler


def _img_residual(ims, ks, overlap):
    ims, stride = int(ims), int(ks - overlap)
    n = ims // stride
    end = n * stride + overlap
    residual = ims - (n * stride) if end > ims else ims % stride
    return residual


def make_patches(image, size, overlap):
    ''' Make patches from the image

    When the image division is not perfect, a zero-padding is performed 
    so that the patches have the same size.

    Returns:
        torch.Tensor:
            patches of shape (B,C,H,W)
    '''

    if isinstance(image, Image.Image):
        image = torchvision.transforms.ToTensor()(image)

    # patches' height & width
    height = min(image.size(1), size[0])
    width = min(image.size(2), size[1])

    # unfold on height
    height_fold = image.unfold(1, height, height - overlap)

    # if non-perfect division on height
    residual = _img_residual(image.size(1), height, overlap)
    if residual != 0:
        # get the residual patch and add it to the fold
        remaining_height = torch.zeros(3, 1, image.size(2), height)  # padding
        remaining_height[:, :, :, :residual] = image[:, -
                                                     residual:, :].permute(0, 2, 1).unsqueeze(1)
        device = height_fold.device
        height_fold = torch.cat(
            (height_fold, remaining_height.to(device)), dim=1)

    # unfold on width
    fold = height_fold.unfold(2, width, width - overlap)

    # if non-perfect division on width, the same
    residual = _img_residual(image.size(2), width, overlap)
    if residual != 0:
        remaining_width = torch.zeros(
            3, fold.shape[1], 1, height, width)  # padding
        remaining_width[:, :, :, :, :residual] = height_fold[:,
                                                             :, -residual:, :].permute(0, 1, 3, 2).unsqueeze(2)
        device = fold.device
        fold = torch.cat((fold, remaining_width.to(device)), dim=2)

    _nrow, _ncol = fold.shape[2], fold.shape[1]

    # reshaping
    patches = fold.permute(1, 2, 0, 3, 4).reshape(-1,
                                                  image.size(0), height, width)
    return patches, (_nrow, _ncol)


def infer_patches(self, patches, batch_size=1):
    dataset = TensorDataset(patches)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset)
    )
    maps = []
    for patch in dataloader:
        patch = patch[0].to(self.device)
        outputs = self.model(patch)
        heatmap = outputs[0]
        scale_factor = 16
        clsmap = F.interpolate(
            outputs[1], scale_factor=scale_factor, mode='nearest')
        outmaps = torch.cat([heatmap, clsmap], dim=1)
        maps = [*maps, *outmaps.unsqueeze(0)]
    return maps


def patch_maps(image, maps, down_ratio, size, overlap, ncol, nrow):

    _, h, w = image.shape
    dh, dw = h // down_ratio, w // down_ratio
    kernel_size = np.array(size) // down_ratio
    stride = kernel_size - overlap // down_ratio
    output_size = (
        ncol * kernel_size[0] - ((ncol-1) * overlap // down_ratio),
        nrow * kernel_size[1] - ((nrow-1) * overlap // down_ratio)
    )

    maps = torch.cat(maps, dim=0)

    n_patches = maps.shape[0]
    maps = maps.permute(1, 2, 3, 0).contiguous().view(1, -1, n_patches)
    out_map = F.fold(maps, output_size=output_size,
                     kernel_size=tuple(kernel_size), stride=tuple(stride))

    out_map = out_map[:, :, 0:dh, 0:dw]

    return out_map


def reduce(self, image, down_ratio, size, overlap,  map):

    dh = image.shape[1] // down_ratio
    dw = image.shape[2] // down_ratio
    ones = torch.ones(image.shape[0], dh, dw)

    ones_patches, n = make_patches(ones,
                                   np.array(size)//down_ratio,
                                   overlap//down_ratio
                                   )
    ncol, nrow = n
    ones_patches = [p.unsqueeze(0).unsqueeze(0)
                    for p in ones_patches[:, 1, :, :]]
    norm_map = patch_maps(image, ones_patches, down_ratio,
                          size, overlap, ncol, nrow)

    return torch.div(map.to(self.device), norm_map.to(self.device))


def local_max(est_map, kernel_size):
    ''' Shape: est_map = [B,C,H,W] '''

    pad = int(kernel_size[0] / 2)
    keep = torch.nn.functional.max_pool2d(
        est_map, kernel_size=kernel_size, stride=1, padding=pad)
    keep = (keep == est_map).float()
    est_map = keep * est_map  # porque nans?

    return est_map


def _get_locs_and_scores(locs_map, scores_map):
    ''' Shapes: locs_map = [H,W] and scores_map = [H,W] '''

    locs_map = locs_map.data.cpu().numpy()
    scores_map = scores_map.data.cpu().numpy()
    locs = []
    scores = []
    for i, j in np.argwhere(locs_map == 1):
        locs.append((i, j))
        scores.append(scores_map[i][j])

    return torch.Tensor(locs), torch.Tensor(scores)


def lmds(est_map, adapt_ts, neg_ts, kernel_size):
    ''' Shape: est_map = [H,W] '''

    est_map_max = torch.max(est_map).item()

    # local maxima
    est_map = local_max(est_map.unsqueeze(0).unsqueeze(0), kernel_size)

    # adaptive threshold for counting
    est_map[est_map < adapt_ts * est_map_max] = 0
    scores_map = torch.clone(est_map)
    est_map[est_map > 0] = 1

    # negative sample
    if est_map_max < neg_ts:
        est_map = est_map * 0

    # count
    count = int(torch.sum(est_map).item())

    # locations and scores
    locs, scores = _get_locs_and_scores(
        est_map.squeeze(0).squeeze(0),
        scores_map.squeeze(0).squeeze(0)
    )

    return count, locs.tolist(), scores.tolist()


def process_maps(heatmap, clsmap, kernel_size=(3, 3), adapt_ts=100.0/255.0, neg_ts=0.1):
    cls_scores = torch.softmax(clsmap, dim=1)[:, 1:, :, :]
    outmaps = torch.cat([heatmap, cls_scores], dim=1)
    batch_size, channels = outmaps.shape[:2]

    b_counts, b_labels, b_scores, b_locs, b_dscores = [], [], [], [], []
    for b in range(batch_size):
        _, locs, _ = lmds(heatmap[b][0], adapt_ts, neg_ts, kernel_size)

        cls_idx = torch.argmax(clsmap[b, 1:, :, :], dim=0)
        classes = torch.add(cls_idx, 1)

        h_idx = torch.Tensor([l[0] for l in locs]).long()
        w_idx = torch.Tensor([l[1] for l in locs]).long()
        labels = classes[h_idx, w_idx].long().tolist()

        chan_idx = cls_idx[h_idx, w_idx].long().tolist()

        scores = cls_scores[b, chan_idx, h_idx, w_idx].float().tolist()

        dscores = heatmap[b, 0, h_idx, w_idx].float().tolist()

        counts = [labels.count(i) for i in range(1, channels)]

        b_labels.append(labels)
        b_scores.append(scores)
        b_locs.append(locs)
        b_counts.append(counts)
        b_dscores.append(dscores)

    # Returning the collected batch-wise results
    return b_counts, b_locs, b_labels, b_scores, b_dscores
