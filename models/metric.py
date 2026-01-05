import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.feature_extraction import create_feature_extractor

import numpy as np
from scipy.stats import entropy
from scipy.linalg import sqrtm
from math import exp

# Cached inception feature extractor
_inception_extractor = None
_inception_device = None

def mae(input, target):
    with torch.no_grad():
        loss = nn.L1Loss()
        output = loss(input, target)
    return output

def psnr(input, target):
    with torch.no_grad():
        mse = torch.mean((input - target) ** 2)
        if mse == 0:
            return float('inf')
        # Assuming images are in [-1, 1], range is 2
        return 20 * torch.log10(2.0 / torch.sqrt(mse))

def ssim(input, target, window_size=11, size_average=True):
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    channel = input.size(1)
    window = create_window(window_size, channel)
    
    if input.is_cuda:
        window = window.cuda(input.get_device())
    window = window.type_as(input)

    mu1 = F.conv2d(input, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(input*input, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target*target, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(input*target, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _get_inception_extractor(device):
    """Lazily create an inception feature extractor returning avgpool features."""
    global _inception_extractor, _inception_device
    if _inception_extractor is None or _inception_device != device:
        model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        model.eval().to(device)
        _inception_extractor = create_feature_extractor(model, {"avgpool": "feat"})
        _inception_device = device
    return _inception_extractor


@torch.no_grad()
def get_inception_features(x):
    """Return 2048-d inception features for images in [-1, 1]."""
    extractor = _get_inception_extractor(x.device)
    x = F.interpolate((x + 1) / 2, size=(299, 299), mode='bilinear', align_corners=False)
    feat = extractor(x)["feat"].squeeze(-1).squeeze(-1)
    return feat


def update_fid_stats(sum_feat, sum_outer, count, feats):
    """Update running sums for FID statistics."""
    sum_feat = sum_feat + feats.sum(dim=0)
    sum_outer = sum_outer + feats.t() @ feats
    count += feats.shape[0]
    return sum_feat, sum_outer, count


def _covariance(sum_feat, sum_outer, count):
    mean = sum_feat / count
    cov = sum_outer / count - torch.ger(mean, mean)
    return mean, cov


def compute_fid_from_stats(sum_r, sum_outer_r, n_r, sum_f, sum_outer_f, n_f):
    """Compute FID given running stats (all tensors on any device)."""
    if n_r == 0 or n_f == 0:
        return float('nan')
    mu1, sigma1 = _covariance(sum_r, sum_outer_r, n_r)
    mu2, sigma2 = _covariance(sum_f, sum_outer_f, n_f)

    mu1 = mu1.cpu().numpy()
    mu2 = mu2.cpu().numpy()
    sigma1 = sigma1.cpu().numpy()
    sigma2 = sigma2.cpu().numpy()

    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)