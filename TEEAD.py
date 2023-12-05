import copy
import json
import os
from pathlib import Path

import numpy as np
import logging
import sys
import platform
import torch
import argparse
import random
import shutil
import signal

from typing import Union, Optional, Sequence, Tuple, Mapping, Hashable, Dict, Any

import monai
from monai import config
from monai.config import KeysCollection, NdarrayOrTensor
from monai.data import *
from monai.data.utils import list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (Compose, LoadImaged, Orientationd, ScaleIntensityd, SaveImaged,
                              EnsureChannelFirstd, Spacingd, AsDiscreted, EnsureTyped, Activationsd, SpatialPadd,
                              KeepLargestConnectedComponentd, SaveImage, AsDiscrete, CenterSpatialCropd, Lambda,
                              AddCoordinateChannelsd, FromMetaTensord, ToMetaTensord, RandFlipd, RandAffined, Lambdad, ToDeviced,
                              MapTransform, Randomizable, RandomizableTransform)
from monai.handlers import (StatsHandler, TensorBoardStatsHandler,
                            CheckpointSaver, CheckpointLoader, ClassificationSaver,
                            ValidationHandler, LrScheduleHandler, SmartCacheHandler)
from monai.networks import predict_segmentation
from monai.networks.nets import *
from monai.networks.layers.simplelayers import Reshape
from monai.networks.layers.convutils import calculate_out_shape, same_padding
from monai.networks.layers import Norm
from monai.networks.blocks import ResidualUnit
from monai.optimizers import Novograd
from monai.losses import DiceLoss, GeneralizedDiceLoss, DiceCELoss
from monai.inferers import SlidingWindowInferer
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.networks.layers.factories import Act

import torch.nn as nn
from monai.utils import issequenceiterable
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from ignite.metrics import Average

import SimpleITK as sitk

# from torchcubicspline import (natural_cubic_spline_coeffs,
#                               NaturalCubicSpline)
#
# import torch_optimizer as optim

from timeit import default_timer as timer


class UNetRegressor(nn.Module):

    def __init__(self, dropout=0.0):
        super(UNetRegressor, self).__init__()

        self.scale_factor = np.sqrt(200)
        self.in_shape = np.array((128, 128, 128))
        self.unet = UNet(dimensions=3, in_channels=1, out_channels=64, channels=(16, 32, 64, 128, 256),
                 strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH, dropout=dropout)
        # self.seg = SegResNet(spatial_dims=3, init_filters=8, in_channels=1, out_channels=10)
        # self.unet = UNETR(1, 64, self.in_shape)
        self.reg = Regressor(in_shape=(64, *self.in_shape), out_shape=[256],
                             channels=(64, 64, 64, 64, 64), strides=(2, 2, 2, 2, 2),
                             norm="INSTANCE", dropout=0.0)
        self.hidden = FullyConnectedNet(256, 63, [128], dropout=0.0)
        self.reshape = Reshape(10,3,2)

    def forward(self, x_in):
        x = self.unet(x_in)
        x = self.reg(x)
        x = self.hidden(x)
        x = torch.tanh(x)
        out = self.reshape(x[:, 3:])

        mags = out[:, :, :, 0] * self.scale_factor
        angles = out[:, :, :, 1] * torch.pi

        real = mags * torch.cos(angles)
        imag = mags * torch.sin(angles)

        out = torch.stack([real, imag], dim=-1)

        centroid = x[:, :3].unsqueeze(1) * self.in_shape[1] * 0.5 * 0.5
        centroid = torch.stack([centroid, torch.zeros_like(centroid)], dim=-1)

        return torch.cat([centroid, out], dim=1)


class ViTRegressor(nn.Module):

    def __init__(self, dropout=0.0):
        super(ViTRegressor, self).__init__()

        self.in_channels = 1
        self.in_shape = np.array((152, 152, 128))
        self.patch_size = (24, 24, 16)
        self.hidden_size = 768
        self.feature_size = np.product(self.in_shape // self.patch_size)
        self.scale_factor = np.sqrt(200)
        self.vit = ViT(in_channels=self.in_channels, img_size=self.in_shape, patch_size=self.patch_size, num_heads=3, mlp_dim=2*self.hidden_size,
                       num_layers=4, hidden_size=self.hidden_size, classification=True, num_classes=256, post_activation=None, dropout_rate=dropout)
        self.final = torch.nn.Sequential(torch.nn.Flatten(),
                                         torch.nn.PReLU(),
                                         FullyConnectedNet(256, 33, [128]),
                                         torch.nn.Tanh())
        self.reshape = Reshape(-1,3,2)

    def forward(self, x_in):
        x, hidden = self.vit(x_in)
        x = self.final(x)

        out = self.reshape(x[:, 3:])

        mags = out[:, :, :, 0] * self.scale_factor
        angles = out[:, :, :, 1] * torch.pi

        real = mags * torch.cos(angles)
        imag = mags * torch.sin(angles)

        out = torch.stack([real, imag], dim=-1)

        centroid = x[:, :3].unsqueeze(1) * self.in_shape[1] * 0.5 * 0.5
        centroid = torch.stack([centroid, torch.zeros_like(centroid)], dim=-1)

        return torch.cat([centroid, out], dim=1)

class AnnulusRegressor(nn.Module):

    def __init__(self):
        super(AnnulusRegressor, self).__init__()

        # From examining training data
        self.scale_factor = 30

        self.in_shape = (1, 152, 152, 128)
        self.channels = (16, 32, 64, 128, 256)
        self.strides = (2, 2, 2, 2, 2)
        self.kernel_size = (3,3,3)
        # self.reg = torch.nn.Sequential(Regressor(in_shape=self.in_shape, out_shape=[256],
        #                               channels=self.channels, strides=self.strides,
        #                               norm="BATCH", dropout=0), torch.nn.PReLU())

        blocks_args_str = [
            "r1_k3_s11_e1_i32_o16_se0.25",
            "r2_k3_s22_e6_i16_o24_se0.25",
            "r2_k5_s22_e6_i24_o40_se0.25",
            "r3_k3_s22_e6_i40_o80_se0.25",
            "r3_k5_s11_e6_i80_o112_se0.25",
            "r4_k5_s22_e6_i112_o192_se0.25",
            "r1_k3_s11_e6_i192_o320_se0.25",
        ]

        self.reg = EfficientNet(blocks_args_str,
                                spatial_dims=3,
                                num_classes=64,
                                in_channels=1,
                                image_size=198,
                                width_coefficient=0.7,
                                depth_coefficient=0.8)

        # self.reg = EfficientNetBN('efficientnet-b0', spatial_dims=3, num_classes=64, in_channels=1, pretrained=False)

        # self.down_path = torch.load(Path(__file__).parent.joinpath('unet_down.md'))

        # for p in self.down_path.parameters():
        #     p.requires_grad = False

        # padding = same_padding(self.kernel_size)
        # self.final_size = np.asarray(self.in_shape[1:], dtype=int)
        # for i, (c, s) in enumerate(zip(self.channels, self.strides)):
        #     self.final_size = calculate_out_shape(self.final_size, self.kernel_size, s, padding)  # type: ignore

        # self.conv = monai.networks.blocks.ResidualUnit(
        #     spatial_dims=3,
        #     in_channels=self.channels[-1],
        #     out_channels=32,
        #     strides=2
        # )
        # self.flatten = torch.nn.Flatten()
        # self.hidden = FullyConnectedNet(np.product(self.final_size) * 32, 33, [128, 64], dropout=0.0)
        self.hidden = FullyConnectedNet(64, 45, [32,16], dropout=0.0, act='memswish')
        self.reshape = Reshape(-1,3,2)

    def forward(self, x_in: torch.Tensor):
        if isinstance(x_in, monai.data.MetaTensor):
            x_in = x_in.as_tensor()

        x = self.reg(x_in)
        # x = self.swish(x)
        # x = self.down_path(x_in)
        # x = self.conv(x)
        # x = self.flatten(x)
        x = self.hidden(x)
        # x = torch.tanh(x)

        out = self.reshape(x[:, 3:])

        mags = torch.sigmoid(out[:,:,:,0]) * self.scale_factor
        angles = torch.tanh(out[:,:,:,1]) * torch.pi

        real = mags * torch.cos(angles)
        imag = mags * torch.sin(angles)

        out = torch.stack([real,imag], dim=-1)

        centroid = torch.tanh(x[:,:3]).unsqueeze(1) * self.in_shape[1] * 0.5 * 0.5
        centroid = torch.stack([centroid, torch.zeros_like(centroid)], dim=-1)

        return torch.cat([centroid, out], dim=1)

class RandSitkEulerTransform(MapTransform, RandomizableTransform):

    def __init__(self,
                 keys: KeysCollection,
                 prob: float = 0.1,
                 rotate_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
                 translate_range: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = None,
                 spacing: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = 0.5,
                 size: Optional[Union[Sequence[Union[Tuple[float, float], float]], float]] = (192, 192, 148),
                 allow_missing_keys: bool = False) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)
        RandomizableTransform.__init__(self, prob)

        self.spacing = monai.utils.misc.ensure_tuple_rep(spacing, 3)
        self.origin = np.array(monai.utils.misc.ensure_tuple_rep(size, 3)) * np.array(spacing) * -0.5

        self.eulerTransform = sitk.Euler3DTransform()
        self.resample = sitk.ResampleImageFilter()
        self.resample.SetOutputSpacing(self.spacing)
        self.resample.SetOutputOrigin(self.origin)
        self.resample.SetTransform(self.eulerTransform)
        self.resample.SetInterpolator(sitk.sitkLinear)
        self.resample.SetSize([192,192,148])

        self.rotate_range = rotate_range
        self.translate_range = translate_range


    def set_random_state(
        self, seed: Optional[int] = None, state: Optional[np.random.RandomState] = None
    ) -> "RandSitkEulerTransform":
        super().set_random_state(seed, state)
        return self

    def _get_rand_param(self, param_range, add_scalar: float = 0.0):
        out_param = []
        for f in param_range:
            if issequenceiterable(f):
                if len(f) != 2:
                    raise ValueError("If giving range as [min,max], should only have two elements per dim.")
                out_param.append(self.R.uniform(f[0], f[1]) + add_scalar)
            elif f is not None:
                out_param.append(self.R.uniform(-f, f) + add_scalar)
        return out_param

    def randomize(self, data: Optional[Any] = None) -> None:
        super().randomize(None)
        if not self._do_transform:
            return None
        self.eulerTransform.SetRotation(*self._get_rand_param(self.rotate_range))
        self.eulerTransform.SetTranslation(self._get_rand_param(self.translate_range))
        self.resample.SetTransform(self.eulerTransform)


    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)

        self.randomize(None)

        if not self._do_transform:
            self.resample.SetTransform(sitk.Euler3DTransform())

        for key in self.key_iterator(d):
            # print(key)
            data = monai.utils.convert_to_numpy(d[key])
            if data.shape[-1] == 3:
                start = timer()
                for i in range(d[key].shape[0]):
                    data[i] = self.eulerTransform.GetInverse().TransformPoint(data[i].tolist())
                d[key] = data
                end = timer()
                # print("Coords: {}".format(end-start))
            else:
                start = timer()
                sim = sitk.GetImageFromArray(data.squeeze().swapaxes(0,2))
                sim_spacing = d[key].pixdim.numpy()
                sim_origin = data.shape[1:] * sim_spacing * -0.5
                sim.SetOrigin(sim_origin)
                sim.SetSpacing(sim_spacing)

                sim_tr = self.resample.Execute(sim)
                sitk.WriteImage(sim_tr, r'D:\pcarnahanfiles\AnnulusTracking\test\test.nii.gz')
                d[key] = np.expand_dims(sitk.GetArrayFromImage(sim_tr).swapaxes(0,2), 0)
                end = timer()
                # print("Image: {}".format(end-start))

        return d





class DEAD:
    keys = ['image']

    spacing = (0.5, 0.5, 0.5)
    size = (192, 192, 148)

    val_tform = Compose([
        LoadImaged(keys),
        EnsureChannelFirstd(keys),
        Spacingd(keys, spacing, align_corners=True, mode='bilinear'),
        Orientationd(keys, axcodes='RAS'),
        ScaleIntensityd("image"),
        SpatialPadd(keys, spatial_size=size),
        CenterSpatialCropd(keys, roi_size=size),
        # AddCoordinateChannelsd("image", (0, 1, 2)),
        EnsureTyped(('image', 'label'), dtype=torch.float),
        # FromMetaTensord('image'),
        ToMetaTensord('image'),
    ])

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    spline_length = 200

    # model = UNetRegressor()
    # model = ViTRegressor()
    model = AnnulusRegressor()

    # model = torch.nn.Sequential(
    #     Regressor(in_shape=(1, 128, 128, 128), out_shape=[512],
    #                                       channels=(16, 32, 64, 128, 256), strides=(2, 2, 2),
    #                                       norm="BATCH", dropout=0.2),
    #     FullyConnectedNet(512, 60, [], dropout=0.2),
    #     Reshape(10,3,2)
    # )

    # model = UNETR(in_channels=1, out_channels=2, img_size=(96,96,96), feature_size=16, hidden_size=768, mlp_dim=3072,
    #               num_heads=12, pos_embed='perceptron', norm_name='instance', dropout_rate=0.2).to(device)

    trainer = None
    logdir = None
    persistent_cache = Path("./persistent_cache")

    @classmethod
    def tform(cls):
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        # device = torch.device('cpu')

        keys = ['image']

        rand_affine = RandAffined(keys, prob=0.25, mode='bilinear', rotate_range=[0.35, 0.35, 0.25],
                                  translate_range=[10,10,6], cache_grid=True, spatial_size=(152,152,128),
                                  padding_mode='zeros')

        rand_flip = RandFlipd(keys, prob=0.5, spatial_axis=0)

        def transform_coords(coords):
            out = monai.utils.convert_to_numpy(coords)

            if rand_flip._do_transform:
                out = out * np.array([-1, 1, 1])

            if rand_affine._do_transform:
                rot = rand_affine.rand_affine.rand_affine_grid.get_transformation_matrix()[:3,:3]
                tr = rand_affine.rand_affine.rand_affine_grid.get_transformation_matrix()[:3,3] * torch.tensor(cls.spacing)
                coords = rot.new_tensor(coords).to(rot.device)
                out = torch.matmul(rot, coords.T)
                out = out.T + tr

            # np.savetxt('./test.csv', out.detach().cpu().numpy())
            print('tform')
            return out

        tform = Compose([
            LoadImaged(keys),
            EnsureChannelFirstd(keys),
            # Spacingd(keys, (0.5, 0.5, 0.5), diagonal=True, align_corners=True, mode='bilinear'),
            Orientationd(keys, axcodes='RAS'),
            ScaleIntensityd("image"),
            # AddCoordinateChannelsd("image", (0, 1, 2)),
            rand_flip,
            rand_affine,
            Lambdad('label', transform_coords),
            Spacingd(keys, cls.spacing, align_corners=True, mode='bilinear'),
            SpatialPadd(keys, spatial_size=cls.size),
            CenterSpatialCropd(keys, roi_size=cls.size),
            # RandSitkEulerTransform(['image', 'label'], 0.25, [0.4, 0.4, 0.25], [8, 8, 5], spacing=cls.spacing, size=cls.size),
            # SaveImaged('image', resample=False),
            EnsureTyped(('image', 'label'), dtype=torch.float),
            # ToDeviced(('image', 'label'), device),
            ToMetaTensord('image'),
            # FromMetaTensord('image')
        ], lazy=True)

        return tform

    @classmethod
    def load_train_data(cls, path, use_val=False):
        train_path = Path(path).joinpath('data').joinpath('train')

        images = [str(p.absolute()) for p in train_path.glob("*US.nii*")]
        segs = [str(p.absolute()) for p in train_path.glob("*annulus.csv")]

        if use_val:
            val_path = Path(path).joinpath('val')
            images += [str(p.absolute()) for p in val_path.glob("*US.nii*")]
            segs += [str(p.absolute()) for p in val_path.glob("*annulus.csv")]

        images.sort()
        segs.sort()

        d = [{"image": im, "label": np.loadtxt(seg)} for im, seg in zip(images, segs)]



        max_size = float('-inf')
        for x in d:
            max_size = max(max_size, x["label"].shape[0])
        cls.spline_length = max_size

        # ds = CacheDataset(d, xform)
        # persistent_cache = cls.persistent_cache
        # persistent_cache.mkdir(parents=True, exist_ok=True)
        # if path == "D:/pcarnahanfiles/AnnulusTracking/":
        #     ds = LMDBDataset(d, cls.tform, cache_dir=persistent_cache, lmdb_kwargs={'writemap': True, 'map_size': 100000000})
        # else:
        #     ds = CacheDataset(d, cls.tform)

        # ds = Dataset(d, cls.tform())
        ds = CacheDataset(d, cls.tform(),
                                num_workers=None,
                                as_contiguous=True,
                                copy_cache=True)
        # ds = PersistentDataset(d, cls.tform, cache_dir=persistent_cache)
        if platform.system() == 'Windows':
            num_workers = 0
        else:
            num_workers = os.cpu_count()

        loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=os.cpu_count(), drop_last=True, pin_memory=False)

        return loader

    @classmethod
    def load_val_data(cls, path, persistent=True, test=False):
        if not test:
            path = Path(path).joinpath('data').joinpath('val')
        else:
            path = Path(path).joinpath('data').joinpath('test')

        random.seed(0)
        images = sorted(str(p.absolute()) for p in path.glob("*US.nii*"))
        segs = sorted([str(p.absolute()) for p in path.glob("*annulus.csv")])
        d = [{"image": im, "label": np.loadtxt(seg)} for im, seg in zip(images, segs)]

        # if persistent:
        #     ds = CacheDataset(d, cls.val_tform)
        # # if persistent and not test:
        # #     persistent_cache = cls.persistent_cache
        # #     persistent_cache.mkdir(parents=True, exist_ok=True)
        # #     ds = LMDBDataset(d, cls.val_tform, cache_dir=persistent_cache, db_name='monai_cache_val', lmdb_kwargs={'map_size': 1000000000})
        # else:
        #     ds = Dataset(d, cls.val_tform)
        ds = Dataset(d, cls.val_tform)

        if platform.system() == 'Windows':
            num_workers = 0
        else:
            num_workers = os.cpu_count()

        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=False)

        return loader

    @classmethod
    def load_eval_data(cls, path):
        path = Path(path)

        random.seed(0)
        images = [str(p.absolute()) for p in path.glob("*.nii*")]
        d = [{"image": im, "label": torch.Tensor()} for im in images]

        # ds = CacheDataset(d, xform)
        ds = Dataset(d, cls.val_tform)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

        return loader

    @classmethod
    def load_model(cls, path, jit=False):
        net_params = torch.load(path, map_location=cls.device)

        if isinstance(net_params, dict):
            cls.model.load_state_dict(net_params['net'])
        elif isinstance(net_params, torch.nn.Module):
            cls.model = net_params
        else:
            logging.error('Could not load network')
            return

        if jit:
            md = cls.model.to(cls.device)
            frozen_mod = torch.jit.optimize_for_inference(torch.jit.trace(md.eval(), torch.randn((1,1,152,152,128), device=cls.device)))
            # jit_md = torch.jit.trace(md, torch.randn((1,1,152,152,128), device=cls.device))
            return frozen_mod

        return cls.model.to(cls.device)

    @classmethod
    def sample_spline_from_fourier(cls, F: torch.Tensor, n: Optional[int] = None):
        """
        Takes Fourier coefficients and applies irfft to sample to length of spline.
        :param batch:
        :param F: Batched Fourier coefficients of annulus. Expected dimensions B x N x 3 x 2
        :return:
        """
        dim = 1

        if not n:
            n = cls.spline_length

        # check if already sampled
        if F.size(dim) == cls.spline_length:
            return F

        if F.size(-1) == 2 and not F.is_complex():
            F = torch.view_as_complex(F.double())

        # c = F[:,0:1,:].real
        # F = torch.cat([torch.zeros_like(F)[:,0:1,:], F[:,1:,:]], dim=dim)

        points = torch.fft.irfft(F, n=n, dim=dim, norm='forward')

        if points.size(-1) == 2:
            theta = torch.linspace(0, 2 * torch.pi, n).to(F.device)
            r = points[:, :, 0].abs()
            polar = points[:, :, 1]

            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            z = polar

            points = torch.stack([x, y, z], dim=-1)

        return points


    @classmethod
    def sample_spline(cls, points):
        if points.size(-1) == 2 or points.dtype == torch.complex64:
            return cls.sample_spline_from_fourier(points)


        n = cls.spline_length
        length, channels = points.shape[1], 3

        if length == n:
            return points

        F = torch.fft.rfft(points, dim=1)
        p2 = torch.fft.irfft(F, n=n, dim=1, norm='forward') / length


        # # Make closed spline by append first point at end
        # if not torch.allclose(points[:,0,:], points[:,-1,:]):
        #     points = torch.cat((points, points[:,0,:].unsqueeze(1)), dim=1)
        #     length += 1
        #
        # t = torch.linspace(0, n, length).floor().long().to(points.device)
        #
        # coeffs = natural_cubic_spline_coeffs(t.float(), points)
        #
        # spline = NaturalCubicSpline(coeffs)
        #
        # t1 = torch.linspace(0, n, n+1).to(points.device)
        #
        # p2 = spline.evaluate(t1)[:, :-1, :]
        return p2

    @classmethod
    def regularized_loss(cls, pred, gt):
        """
        Training loss function with regularization
        :param pred: Tensor with shape bx6x3x2 as Fourier representation of annulus in each of 3 dimensions. 6 Fourier coeficcients x 3 dimensions x 2 (complex valued)
        :param gt: Tensor in form bx200x3, points of annulus ring. Normalized coordinates relative to center of image
        :return: Scalar loss
        """

        # Dimension to reduce along
        dim = 1 if pred.dim() >= 4 else 0


        # Get ground truth as Foruier coefficients
        # gt_F = torch.fft.rfft(gt, dim=dim, norm='forward')
        # Resample ground truth with low pass filter
        # gt_resampled = cls.sample_spline_from_fourier(gt_F[:, :11], 30)

        # Sample 200 annulus points from pred
        pred_sampled = cls.sample_spline_from_fourier(pred)
        # Sample 30 points from pred
        pred_sampled2 = cls.sample_spline_from_fourier(pred, 30)

        # Get centroids of pred and gt
        # pred_centroid = pred[:,0,:,0]
        pred_centroid = pred_sampled.mean(axis=dim)
        gt_centroid = gt.mean(axis=dim)

        # Compute error between centroids
        # centroid_dist = torch.linalg.vector_norm(pred_centroid - gt_centroid, dim=dim).mean()
        centroid_dist = torch.nn.functional.smooth_l1_loss(pred_centroid, gt_centroid, beta=10.0)
        # centroid_dist = torch.log(centroid_dist + 1).pow(2)

        # Get distances between all adjacent points
        pdist = torch.linalg.vector_norm(pred_sampled.roll(1, dims=dim) - pred_sampled, dim=dim + 1)
        pdist_gt = torch.linalg.vector_norm(gt.roll(1, dims=dim) - gt, dim=dim + 1)

        # Circumference difference between gt and pred (sum of differences between adjacent points)
        # circum = (pdist.sum(dim=dim) - pdist_gt.sum(dim=dim)).abs().mean()
        pred_perim = pdist.sum(dim=dim)
        gt_perim = pdist_gt.sum(dim=dim)
        circum = torch.nn.functional.smooth_l1_loss(pred_perim, gt_perim, beta=20.0)

        # mean absolute error calculations
        # mae = cls.mae_spline_loss(pred_sampled, gt)
        # mae = torch.linalg.vector_norm(pred_sampled - gt, dim=dim + 1).mean()

        # Distance between corresponding points in pred and gt (requires them to start at same place)
        # dist = torch.linalg.vector_norm(pred_sampled2 - gt_resampled, dim=-1).mean()

        # Get distances between all points pairs
        d = torch.cdist(gt.double(), pred_sampled.double())
        d_min1 = d.min(dim=2)[0]
        d_min2 = d.min(dim=1)[0]

        # Average symmetric loss
        d_min = torch.cat([d_min1, d_min2], dim=1)
        # mae = torch.log(d_min.pow(2).mean() + 1).pow(2)
        mae = torch.nn.functional.smooth_l1_loss(d_min, torch.zeros_like(d_min), beta=5.0)

        # Sum of angles of pred in xy plane (should complete 1 circle)
        a = pred_sampled[:, :, :2]
        b = a.roll(1, dims=dim)
        inner_product = (a * b).sum(dim=dim + 1)
        a_norm = torch.linalg.vector_norm(a, dim=dim + 1)
        b_norm = torch.linalg.vector_norm(b, dim=dim + 1)
        angle = torch.acos(inner_product / (a_norm * b_norm))
        torch.sum(angle, dim=dim)
        angle_loss = torch.mean(torch.abs(2*torch.pi - torch.sum(angle, dim=dim)))

        # Attempt to prevent network spitting out large values
        a = torch.linspace(0.01, 0.1, pred.size(1) - 1, device=d.device).pow(2)
        fft_reg = (torch.view_as_complex(pred.double()).abs()[:, 1:] * a.unsqueeze(1)).mean()
        # fft_reg = fft_reg if fft_reg > 1 else 0

        # Create distance map from point coordinates
        def pos2dis(y):

            b = y.size(0)

            n = torch.linspace(-38, 38, 54).float().to(y.device)
            n1 = torch.linspace(-32, 32, 42).float().to(y.device)
            m = torch.stack(torch.meshgrid([n, n, n1], indexing='ij'), dim=-1)

            s = m.shape[:-1]

            idx = torch.zeros((b, *s), device=m.device).long()
            for i in range(s[-1]):
                idx[:, :, :, i] = torch.cdist(y.float(),
                                              m[:, :, i].reshape(-1, 3).float()).min(dim=1)[1].reshape(b, *s[:-1])

            idx = idx.reshape(b, -1)
            batch_index = torch.arange(b).view(b, 1).expand_as(idx)

            dis = torch.linalg.vector_norm(m.reshape(-1, 3).expand(b, -1, -1) - y[batch_index, idx], dim=-1).reshape(b,
                                                                                                                     *s)
            dis = torch.nn.functional.interpolate(dis.unsqueeze(1), scale_factor=2, mode='trilinear',
                                                  align_corners=True).squeeze()

            return dis

        # cd = pos2dis(pred_sampled)
        # cdr = cd.subtract(4).mul(-20)

        # n = torch.linspace(-30, 30, 76).double().to(pred.device)
        # n1 = torch.linspace(-25, 25, 56).double().to(pred.device)
        # m = torch.stack(torch.meshgrid([n, n, n1], indexing='ij'), dim=-1)
        # idx = torch.cdist(m.reshape(-1, 3), pred_sampled).min(dim=-1)[1]
        # batch_index = torch.arange(b).view(b, 1).expand_as(idx)
        # cd = torch.linalg.vector_norm(m.reshape(b, -1, 3) - pred_sampled[batch_index, idx], dim=-1)
        # cdr = cd.reshape(b, 76, 76, 56).subtract(3).mul(-20)

        # gt_cd = pos2dis(gt)
        # cdrb = gt_cd.sub(4).mul(-1).sign().add(1).div(2)
        #
        # l = monai.losses.DiceLoss(sigmoid=True)(cdr, cdrb) + torch.nn.BCEWithLogitsLoss()(cdr, cdrb)

        # print("MAE: {}, Centroid: {}, circum: {}, fft_reg: {}, AngleLoss: {}".format(mae, centroid_dist, circum, fft_reg, angle_loss))

        # if cls.trainer.state.epoch < 50:
        #     return mae + fft_reg
        # else:
        #     return mae + angle_loss + fft_reg + centroid_dist
        # return 2 * mae + angle_loss + 0.5*centroid_dist + 0.1*circum + fft_reg

        return 2 * mae + 0.5 * centroid_dist + 0.1 * circum

    @classmethod
    def mae_spline_loss(cls, pred, p1):
        # n = p1.shape[1]
        # length, channels = pred.shape[1], 3
        # t = torch.linspace(0, n - 1, length).floor().long().to(p1.device)
        #
        # coeffs = natural_cubic_spline_coeffs(t.float(), pred)
        #
        # spline = NaturalCubicSpline(coeffs)
        #
        # t1 = torch.linspace(0, n - 1, n).to(p1.device)

        p2 = cls.sample_spline(pred)
        d = torch.cdist(p1.double(), p2.double())
        d_min1 = d.min(dim=2)[0]
        d_min2 = d.min(dim=1)[0]

        # Average symmetric loss
        d_min = torch.cat([d_min1, d_min2], dim=1)

        return (d_min.abs()).mean()

    # @classmethod
    # def convert_normalized_coord_to_mm(cls, points, meta_dict):
    #     if points.max().item() > 1.5:
    #         return points
    #
    #     extent = torch.nn.functional.pad(meta_dict['spatial_shape'] - 1, (0, 1), "constant", 1.0).to(points.device)
    #     orig_affine = meta_dict['original_affine'].to(points.device)
    #
    #     lb = torch.matmul(orig_affine, torch.tensor([0,0,0,1], dtype=torch.float64, device=points.device))
    #     ub = torch.matmul(orig_affine, extent.double())
    #
    #     slopes = 1 / (ub - lb)
    #     out = (points.double() + 0.5) / slopes[0:3]
    #
    #     return out


    @classmethod
    def output_tform(cls, x):
        # pred = []
        # label = []
        # for item in x:
        #     pred.append(cls.convert_normalized_coord_to_mm(cls.sample_spline(item['pred']), item['image'].meta))
        #     label.append(cls.convert_normalized_coord_to_mm(item['label'], item['image'].meta))

        xt = list_data_collate(x)
        return cls.mae_spline_loss(xt['pred'], xt['label'])

    @classmethod
    def train(cls, data=None, use_val=False, load_checkpoint=None):
        config.print_config()

        if not data:
            dataPath = "D:/pcarnahanfiles/AnnulusTracking/"
        else:
            dataPath = data

        cls.persistent_cache = Path(dataPath).joinpath('persistent_cache')
        cls.persistent_cache.mkdir(parents=True, exist_ok=True)

        loader = cls.load_train_data(dataPath, use_val=use_val)

        net = cls.model.to(cls.device)

        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print("Number of trainable parameters: {}".format(trainable_params))

        # clip_value = 5
        # for p in net.parameters():
        #     p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

        # loss = DiceCELoss(softmax=True, include_background=False, lambda_dice=0.5)

        opt = Novograd(net.parameters(), 1e-2)
        # opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        # opt = torch.optim.SGD(net.parameters(), lr=1e-2)
        # opt = torch.optim.AdamW(net.parameters(), lr=1e-3)

        # trainer = create_supervised_trainer(net, opt, loss, device, False, )
        trainer = SupervisedTrainer(
            device=cls.device,
            max_epochs=1000,
            train_data_loader=loader,
            network=net,
            optimizer=opt,
            loss_function=cls.regularized_loss,
            # postprocessing=cls.output_tform,
            # decollate=False,
            key_train_metric={"train_MAE (mm)": Average(output_transform=cls.output_tform)},
            metric_cmp_fn=lambda curr, prev: curr < prev if prev > 0 else True,
            amp=True
        )
        cls.trainer = trainer

        # Load checkpoint if defined
        if load_checkpoint:
            checkpoint = torch.load(load_checkpoint)
            if checkpoint['trainer']:
                trainer.state.epoch = checkpoint['trainer']['iteration'] // checkpoint['trainer']['epoch_length']
                trainer.state.iteration = checkpoint['trainer']['iteration']
                # for k in checkpoint['trainer']:
                #     trainer.state.__dict__[k] = checkpoint['trainer'][k]
                # trainer.state.epoch = trainer.state.iteration // trainer.state.epoch_length
            checkpoint_loader = CheckpointLoader(load_checkpoint, {'net': net, 'opt': opt})
            checkpoint_loader.attach(trainer)

            logdir = Path(load_checkpoint).parent

        else:
            logdir = Path(dataPath).joinpath('./runs/')
            logdir.mkdir(parents=True, exist_ok=True)
            dirs = sorted([int(x.name) for x in logdir.iterdir() if x.is_dir()])
            if not dirs:
                logdir = logdir.joinpath('0')
            else:
                logdir = logdir.joinpath(str(int(dirs[-1]) + 1))
            logdir.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(Path(__file__)), str(logdir.joinpath('TEEAD.py')))

        cls.logdir = logdir


        # Adaptive learning rate
        lr_scheduler = StepLR(opt, 500, 0.5)
        lr_handler = LrScheduleHandler(lr_scheduler)
        lr_handler.attach(trainer)

        ### optional section for checkpoint and tensorboard logging
        # adding checkpoint handler to save models (network params and optimizer stats) during training
        checkpoint_handler = CheckpointSaver(logdir, {'net': net, 'opt': opt, 'trainer': trainer}, n_saved=4, save_final=True,
                                             epoch_level=True, save_interval=100)
        checkpoint_handler.attach(trainer)

        # StatsHandler prints loss at every iteration and print metrics at every epoch
        train_stats_handler = StatsHandler(
            name='trainer',
            output_transform=lambda x: x[0]['loss'])
        train_stats_handler.attach(trainer)

        test = monai.utils.first(loader)['image']

        tb_writer = SummaryWriter(log_dir=logdir)
        # tb_writer.add_graph(net, monai.utils.first(loader)['image'].to(cls.device))

        # TensorBoardStatsHandler plots loss at every iteration and plots metrics at every epoch, same as StatsHandler
        train_tensorboard_stats_handler = TensorBoardStatsHandler(
            summary_writer=tb_writer,
            output_transform=lambda x: x[0]['loss'],
            tag_name='loss')
        train_tensorboard_stats_handler.attach(trainer)

        # Set up validation step
        val_loader = cls.load_val_data(dataPath)

        evaluator = SupervisedEvaluator(
            device=cls.device,
            val_data_loader=val_loader,
            network=net,
            # prepare_batch= lambda batch, device, non_blocking: (batch['image'].as_tensor().to(device), batch['label'].to(device)),
            # decollate=False,
            key_val_metric={"val_MAE (mm)": Average(output_transform=cls.output_tform)},
            metric_cmp_fn=lambda curr, prev: curr < prev if prev > 0 else True,
        )

        val_stats_handler = StatsHandler(
            name='evaluator',
            output_transform=lambda x: None,  # no need to plot loss value, so disable per iteration output
            global_epoch_transform=lambda x: trainer.state.epoch)  # fetch global epoch number from trainer
        val_stats_handler.attach(evaluator)

        # add handler to record metrics to TensorBoard at every validation epoch
        val_tensorboard_stats_handler = TensorBoardStatsHandler(
            summary_writer=tb_writer,
            output_transform=lambda x: None,  # no need to plot loss value, so disable per iteration output
            global_epoch_transform=lambda x: trainer.state.epoch)  # fetch global epoch number from trainer
        val_tensorboard_stats_handler.attach(evaluator)

        val_handler = ValidationHandler(
            validator=evaluator,
            interval=10
        )
        val_handler.attach(trainer)

        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        trainer.run()

        torch.save(net, logdir.joinpath('final_model.md'))

    @classmethod
    def validate(cls, load_checkpoint, data=None, use_test=False):
        config.print_config()

        if not data:
            loader = cls.load_val_data("D:/pcarnahanfiles/AnnulusTracking/", False, test=use_test)
        else:
            loader = cls.load_val_data(data, False, test=use_test)

        net = cls.load_model(load_checkpoint, False)

        logdir = Path(load_checkpoint).parent.joinpath('out')
        if not logdir.exists():
            logdir.mkdir()

        batch = monai.data.utils.first(loader)

        # torch.onnx.export(
        #     net.eval(),
        #     batch['image'].as_tensor().to(cls.device),
        #     str(logdir.joinpath('net.onnx')),
        #     export_params=True,  # store the trained parameter weights inside the model file
        #     do_constant_folding=True,  # whether to execute constant folding for optimization
        #     opset_version=14,
        #     operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
        #     input_names=["images"],  # the model's input names
        #     output_names=['output'],  # the model's output names
        #     dynamic_axes={'images': {0: 'batch_size'},  # variable length axes
        #                   'output': {0: 'batch_size'}})

        start = timer()
        avg = Average(device=cls.device)
        net.eval()
        with torch.no_grad():
            for batch in loader:
                im = batch['image'].to(cls.device)
                p1 = batch['label'].to(cls.device)

                with torch.cuda.amp.autocast():
                    p2 = net(im)

                batch['pred'] = cls.sample_spline(p2)

                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(cls.device)

                mm_mae = cls.output_tform(decollate_batch(batch))
                avg.update(mm_mae)
                print("error (mm): {}".format(mm_mae))

                out = decollate_batch(batch['pred'])
                labels = decollate_batch(batch['label'])
                meta_dict = decollate_batch(im.meta)
                for o, l, m in zip(out, labels, meta_dict):
                    f = logdir.joinpath(Path(m['filename_or_obj']).stem + '_pred.csv')
                    f_gt = logdir.joinpath(Path(m['filename_or_obj']).stem + '_gt.csv')
                    np.savetxt(str(f_gt), l.detach().cpu().numpy())
                    np.savetxt(str(f), o.detach().cpu().numpy())

        mean_mae = avg.compute()
        end = timer()


        print("Metric Mean_MAE: {}".format(mean_mae.item()))
        print("Elapsed time {}s".format(end-start))

    @classmethod
    def evaluate(cls, load_checkpoint, data):
        config.print_config()

        loader = cls.load_eval_data(data)

        device = cls.device
        net = cls.load_model(load_checkpoint, True)

        logdir = Path(data).joinpath('out')
        if not logdir.exists():
            logdir.mkdir(parents=True, exist_ok=True)


        net.eval()
        with torch.no_grad():
            for batch in loader:
                im = batch['image'].to(cls.device)

                p2 = net(im)

                out = decollate_batch(cls.sample_spline(p2))
                meta_dict = decollate_batch(batch['image_meta_dict'])
                for o, m in zip(out, meta_dict):
                    # pred_mm = cls.convert_normalized_coord_to_mm(o, m)
                    f = logdir.joinpath(Path(m['filename_or_obj']).stem + '_pred.csv')
                    # fmm = logdir.joinpath(Path(m['filename_or_obj']).stem + '_pred_mm.csv')
                    np.savetxt(str(f), o.detach().cpu().numpy())
                    # np.savetxt(str(fmm), pred_mm.detach().cpu().numpy())

    @classmethod
    def closed_curve_markup(cls, data):

        with open(str(Path(__file__).parent.joinpath('CC.mrk.json'))) as f:
            mk = json.load(f)

        point = copy.deepcopy(mk['markups'][0]['controlPoints'][0])
        point['visibility'] = True

        mk['markups'][0]['controlPoints'] = []
        mk['markups'][0]['type'] = 'ClosedCurve'

        # Blue
        mk['markups'][0]['display']['color'] = [0.0, 0.0, 0.5]
        mk['markups'][0]['display']['selectedColor'] = [0.0, 0.0, 0.5]

        for i, row in enumerate(data):
            point['id'] = str(i)
            point['position'] = row.tolist()

            mk['markups'][0]['controlPoints'].append(point)
            mk['markups'][0]['lastUsedControlPointNumber'] = str(i)

        return mk

    @classmethod
    def handle_sigint(cls, signum, frame):
        if cls.trainer:
            msg = "Ctrl-c was pressed. Stopping run at epoch {}.".format(cls.trainer.state.epoch)
            cls.trainer.should_terminate = True
            cls.trainer.should_terminate_single_epoch = True
        else:
            msg = "Ctrl-c was pressed. Stopping run."
            print(msg, flush=True)
            exit(1)
        print(msg, flush=True)




def parse_args():
    parser = argparse.ArgumentParser(description='DEAD training')

    subparsers = parser.add_subparsers(help='sub-command help', dest='mode')
    subparsers.required = True

    train_parse = subparsers.add_parser('train', help="Train the network")
    train_parse.add_argument('-load', type=str, help='load from a given checkpoint')
    train_parse.add_argument('-data', type=str, help='data folder. should contain "train" and "val" sub-folders')
    train_parse.add_argument('-use_val', action='store_true', help='Flag to indicate that training set should include validation data.')

    val_parse = subparsers.add_parser('validate', help='Evaluate the network')
    val_parse.add_argument('load', type=str, help='load from a given checkpoint')
    val_parse.add_argument('-data', type=str, help='data folder. should contain "train" and "val" sub-folders')
    val_parse.add_argument('-use_test', action='store_true',
                             help='Run on test data')

    seg_parse = subparsers.add_parser('evaluate', help='Evaluate on images')
    seg_parse.add_argument('load', type=str, help='load from a given checkpoint')
    seg_parse.add_argument('data', type=str, help='data folder')

    return parser.parse_args()

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    signal.signal(signal.SIGINT, DEAD.handle_sigint)
    start = timer()
    args = parse_args()
    if args.mode == 'validate':
        DEAD.validate(args.load, args.data, args.use_test)
    elif args.mode == 'train':
        DEAD.train(args.data, args.use_val, args.load)
    elif args.mode == 'evaluate':
        DEAD.evaluate(args.load, args.data)
    end = timer()
    print({"Total runtime: {}".format(end - start)})


if __name__ == "__main__":
    main()