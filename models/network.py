import math
import torch
import torch.nn as nn
import kornia
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from core.base_network import BaseNetwork
class Network(BaseNetwork):
    def __init__(self, unet, beta_schedule, module_name='sr3', **kwargs):
        init_type = kwargs.get('init_type', 'kaiming')
        gain = kwargs.get('gain', 0.02)
        super(Network, self).__init__(init_type=init_type, gain=gain)

        if module_name == 'sr3':
            from .sr3_modules.unet import UNet
        elif module_name == 'guided_diffusion':
            from .guided_diffusion_modules.unet import UNet
        
        self.denoise_fn = UNet(**unet)
        self.beta_schedule = beta_schedule
        # Bai & Xu: 1x1 Convolution for Feature Extraction
        self.feature_extractor = nn.Conv2d(3, 3, kernel_size=1)
        
        # Hyper-parameter for Color Loss
        self.color_loss_weight = kwargs.get('color_loss_weight', 0.1)

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule[phase])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def predict_start_from_noise_with_gammas(self, y_t, gammas, noise):
        sqrt_recip_gammas = torch.sqrt(1.0 / gammas)
        sqrt_recipm1_gammas = torch.sqrt(1.0 / gammas - 1.0)
        return sqrt_recip_gammas * y_t - sqrt_recipm1_gammas * noise

    def q_posterior(self, y_0_hat, y_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y_0_hat = self.predict_start_from_noise(
                y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level))

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None):
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t>0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8):
        b, *_ = y_cond.shape

        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps//sample_num)
        
        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps, leave=False):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, y_cond=y_cond)
            if mask is not None:
                y_t = y_0*(1.-mask) + mask*y_t
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
        return y_t, ret_arr

    def get_co_weights(self, y_cond, co_lambda=2.0):
        # CPTrans: Content-Aware Optimization
        
        # 0. Pre-smoothing to suppress SAR speckle noise
        # Using a 5x5 Gaussian kernel with sigma=1.5
        y_cond_smooth = kornia.filters.gaussian_blur2d(y_cond, (5, 5), (1.5, 1.5))
        
        # 1. Compute spatial gradient on smoothed image
        gradients = kornia.filters.spatial_gradient(y_cond_smooth)
        
        # 2. Compute magnitude: sqrt(dx^2 + dy^2)
        dx = gradients[:, :, 0, :, :]
        dy = gradients[:, :, 1, :, :]
        magnitude = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
        
        # Average magnitude across channels if input is multi-channel
        if magnitude.shape[1] > 1:
            magnitude = magnitude.mean(dim=1, keepdim=True)
            
        # 3. Normalize magnitude to [0, 1] (Per-Instance Normalization)
        # Flatten spatial dimensions to find min/max per image in batch
        b, c, h, w = magnitude.shape
        mag_flat = magnitude.view(b, -1)
        mag_min = mag_flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        mag_max = mag_flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
        
        normalized_magnitude = (magnitude - mag_min) / (mag_max - mag_min + 1e-6)
        
        # 4. Compute weights
        weights = 1.0 + co_lambda * normalized_magnitude
        return weights.detach()

    def forward(self, y_0, y_cond=None, mask=None, noise=None):
        # sampling from p(gammas)
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
        sqrt_gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (sqrt_gamma_t2-gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        # Bai & Xu: Feature Extraction with 1x1 Conv
        fm = self.feature_extractor(y_cond)

        if mask is not None:
            noise_hat = self.denoise_fn(torch.cat([fm, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas)
            loss = self.loss_fn(mask*noise, mask*noise_hat)
        else:
            # Bai & Xu: Concatenate Feature Map
            noise_hat = self.denoise_fn(torch.cat([fm, y_noisy], dim=1), sample_gammas)
            # CPTrans（保持关闭）或原始噪声损失
            if self.training and False:
                weights = self.get_co_weights(y_cond)
                loss_pixel = (noise - noise_hat) ** 2
                loss_weighted = loss_pixel * weights
                loss = loss_weighted.mean()
            else:
                loss = self.loss_fn(noise, noise_hat)

            # 颜色监督损失：使用基于 sample_gammas 的精确反推
            y_0_hat = self.predict_start_from_noise_with_gammas(
                y_noisy,
                sample_gammas.view(-1, 1, 1, 1),
                noise_hat
            )
            # 数值稳定性
            y_0_hat = torch.clamp(y_0_hat, -1.0, 1.0)

            kernel_size = (5, 5)
            sigma = (0.8, 0.8)
            blurred_pred = kornia.filters.gaussian_blur2d(y_0_hat, kernel_size, sigma)
            blurred_gt = kornia.filters.gaussian_blur2d(y_0, kernel_size, sigma)

            loss_color = torch.nn.functional.mse_loss(blurred_pred, blurred_gt)
            loss = loss + self.color_loss_weight * loss_color

        return loss


# gaussian diffusion trainer class
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract(a, t, x_shape=(1,1,1,1)):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# beta_schedule function
def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas

def make_beta_schedule(schedule, n_timestep, linear_start=1e-6, linear_end=1e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


