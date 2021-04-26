from fastai.callback.hook import hook_outputs
from fastai.layers import ConvLayer, PixelShuffle_ICNR, BatchNorm, SelfAttention, ResBlock, NormType, flatten_model, \
    MergeLayer, SigmoidRange, SequentialEx, Lambda
from fastai.torch_core import Module, apply_init, init_default, one_param
from fastai.vision.models import resnet34
import torch
from fastai.vision.models.unet import UnetBlock
from fastcore.meta import delegates
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm
import numpy as np


def relu(inplace: bool = False, leaky: float = None):
    "Return a relu activation, maybe `leaky` and `inplace`."
    return nn.LeakyReLU(inplace=inplace, negative_slope=leaky) if leaky is not None else nn.ReLU(inplace=inplace)


def conv_layer(ni: int, nf: int, ks: int = 3, stride: int = 1, padding: int = None, bias: bool = None,
               is_1d: bool = False,
               norm_type=NormType.Batch, use_activ: bool = True, leaky: float = None,
               transpose: bool = False, init=nn.init.kaiming_normal_, self_attention: bool = False):
    if padding is None: padding = (ks - 1) // 2 if not transpose else 0
    bn = norm_type in (NormType.Batch, NormType.BatchZero)
    if bias is None: bias = not bn
    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d
    conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding), init)
    if norm_type == NormType.Weight:
        conv = weight_norm(conv)
    elif norm_type == NormType.Spectral:
        conv = spectral_norm(conv)
    layers = [conv]
    if use_activ: layers.append(relu(True, leaky=leaky))
    if bn: layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm2d)(nf))
    if self_attention: layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)


def model_sizes(m: nn.Module, size: tuple = (64, 64)):
    "Pass a dummy input through the model `m` to get the various sizes of activations."
    with hook_outputs(m) as hooks:
        x = dummy_eval(m, size)
        return [o.stored.shape for o in hooks]


def _get_sfs_idxs(sizes):
    "Get the indexes of the layers where the size of the activation changes."
    feature_szs = [size[-1] for size in sizes]
    sfs_idxs = list(np.where(np.array(feature_szs[:-1]) != np.array(feature_szs[1:]))[0])
    if feature_szs[0] != feature_szs[1]: sfs_idxs = [0] + sfs_idxs
    return sfs_idxs


def in_channels(m: nn.Module):
    "Return the shape of the first weight layer in `m`."
    for l in flatten_model(m):
        if hasattr(l, 'weight'):
            return l.weight.shape[1] * l.groups if hasattr(l, 'groups') else l.weight.shape[1]

    raise Exception('No weight layer')


def dummy_batch(m: nn.Module, size: tuple = (64, 64)):
    "Create a dummy batch to go through `m` with `size`."
    ch_in = in_channels(m)
    return one_param(m).new(1, ch_in, *size).requires_grad_(False).uniform_(-1., 1.)


def dummy_eval(m: nn.Module, size: tuple = (64, 64)):
    "Pass a `dummy_batch` in evaluation mode in `m` with `size`."
    return m.eval()(dummy_batch(m, size))


def res_block(nf, dense: bool = False, norm_type=NormType.Batch, bottle: bool = False, **conv_kwargs):
    "Resnet block of `nf` features. `conv_kwargs` are passed to `conv_layer`."
    norm2 = norm_type
    if not dense and (norm_type == NormType.Batch): norm2 = NormType.BatchZero
    nf_inner = nf // 2 if bottle else nf
    return SequentialEx(conv_layer(nf, nf_inner, norm_type=norm_type, **conv_kwargs),
                        conv_layer(nf_inner, nf, norm_type=norm2, **conv_kwargs),
                        MergeLayer(dense))


def batchnorm_2d(nf: int, norm_type: NormType = NormType.Batch):
    "A batchnorm2d layer with `nf` features initialized depending on `norm_type`."
    bn = nn.BatchNorm2d(nf)
    with torch.no_grad():
        bn.bias.fill_(1e-3)
        bn.weight.fill_(0. if norm_type == NormType.BatchZero else 1.)
    return bn
