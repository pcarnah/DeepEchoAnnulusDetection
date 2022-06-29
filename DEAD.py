from pathlib import Path

import numpy as np
import logging
import sys
import torch
import argparse
import random

import monai
from monai import config
from monai.data import Dataset, DataLoader, GridPatchDataset, CacheDataset, PersistentDataset, LMDBDataset
from monai.data.utils import list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (Compose, LoadImaged, Orientationd, ScaleIntensityd,
                              EnsureChannelFirstd, Spacingd, AsDiscreted, EnsureTyped, Activationsd, SpatialPadd,
                              KeepLargestConnectedComponentd, SaveImage, AsDiscrete, CenterSpatialCropd,
                              AddCoordinateChannelsd, ToMetaTensord)
from monai.handlers import (StatsHandler, TensorBoardStatsHandler,
                            CheckpointSaver, CheckpointLoader, ClassificationSaver,
                            ValidationHandler, LrScheduleHandler)
from monai.networks import predict_segmentation
from monai.networks.nets import *
from monai.networks.layers import Norm
from monai.optimizers import Novograd
from monai.losses import DiceLoss, GeneralizedDiceLoss, DiceCELoss
from monai.inferers import SlidingWindowInferer
from monai.engines import SupervisedTrainer, SupervisedEvaluator

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from ignite.metrics import Average

from torchcubicspline import (natural_cubic_spline_coeffs,
                              NaturalCubicSpline)

from timeit import default_timer as timer

class DEAD:

    keys = ("image")
    tform = Compose([
        LoadImaged(keys),
        EnsureChannelFirstd(keys),
        Spacingd(keys, (0.5, 0.5, 0.5), diagonal=True, mode='bilinear'),
        Orientationd(keys, axcodes='RAS'),
        ScaleIntensityd("image"),
        AddCoordinateChannelsd("image", (0, 1, 2)),
        SpatialPadd(keys, spatial_size=(128,128,128)),
        CenterSpatialCropd(keys, roi_size=(128,128,128)),
        EnsureTyped(('image', 'label'), dtype=torch.float),
        ToMetaTensord('image')
    ])

    # val_tform = Compose([
    #     LoadImaged(keys),
    #     EnsureChannelFirstd(keys),
    #     Spacingd(keys, (0.5, 0.5, 0.5), diagonal=True, mode='bilinear'),
    #     Orientationd(keys, axcodes='RAS'),
    #     ScaleIntensityd("image"),
    #     AddCoordinateChannelsd("image", (1, 2, 3)),
    #     SpatialPadd(keys, spatial_size=(128, 128, 128)),
    #     CenterSpatialCropd(keys, roi_size=(128, 128, 128)),
    #     EnsureTyped(('image', 'label')),
    # ])
    #
    # seg_tform = Compose([
    #     LoadImaged('image'),
    #     EnsureChannelFirstd('image'),
    #     Spacingd('image', (0.5, 0.5, 0.5), diagonal=True, mode='bilinear'),
    #     Orientationd('image', axcodes='RAS'),
    #     ScaleIntensityd("image"),
    #     AddCoordinateChannelsd("image", (1, 2, 3)),
    #     SpatialPadd(keys, spatial_size=(128, 128, 128)),
    #     CenterSpatialCropd(keys, roi_size=(128, 128, 128)),
    #     EnsureTyped('image'),
    # ])

    post_tform = Compose(
        [Activationsd(keys='pred', softmax=True),
         AsDiscreted(keys='pred', argmax=True, to_onehot=True, n_classes=2),
         KeepLargestConnectedComponentd(keys='pred', applied_labels=1)
         # AsDiscreted(keys=('label','pred'), to_onehot=True, n_classes=2),
        ]
    )

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    spline_length = 200

    model = Regressor(in_shape=(4,128,128,128),out_shape=(15,3),
                                          channels=(16,32,64,128,256), strides=(2,2,2),
                                          norm="BATCH").to(device)

    # model = UNETR(in_channels=1, out_channels=2, img_size=(96,96,96), feature_size=16, hidden_size=768, mlp_dim=3072,
    #               num_heads=12, pos_embed='perceptron', norm_name='instance', dropout_rate=0.2).to(device)

    @classmethod
    def load_train_data(cls, path, use_val=False):
        train_path = Path(path).joinpath('train')

        images = [str(p.absolute()) for p in train_path.glob("*US.nii")]
        segs = [str(p.absolute()) for p in train_path.glob("*annulus.csv")]

        if use_val:
            val_path = Path(path).joinpath('val')
            images += [str(p.absolute()) for p in val_path.glob("*US.nii")]
            segs += [str(p.absolute()) for p in val_path.glob("*annulus.csv")]

        images.sort()
        segs.sort()

        d = [{"image": im, "label": np.loadtxt(seg)} for im, seg in zip(images, segs)]

        max_size = float('-inf')
        for x in d:
            max_size = max(max_size, x["label"].shape[0])
        cls.spline_length = max_size

        # ds = CacheDataset(d, xform)
        persistent_cache = Path("./persistent_cache")
        persistent_cache.mkdir(parents=True, exist_ok=True)
        ds = LMDBDataset(d, cls.tform, cache_dir=persistent_cache, lmdb_kwargs={'map_size': 3e9})
        # ds = Dataset(d, cls.tform)
        # ds = CacheDataset(d, cls.tform)
        loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, drop_last=True, pin_memory=torch.cuda.is_available())

        return loader

    @classmethod
    def load_val_data(cls, path, persistent=True, test=False):
        if not test:
            path = Path(path).joinpath('val')
        else:
            path = Path(path).joinpath('test')

        random.seed(0)
        images = sorted(str(p.absolute()) for p in path.glob("*US.nii"))
        segs = [str(p.absolute()) for p in path.glob("*annulus.csv")]
        d = [{"image": im, "label": np.loadtxt(seg)} for im, seg in zip(images, segs)]

        # ds = CacheDataset(d, xform)
        if persistent and not test:
            persistent_cache = Path("./persistent_cache")
            persistent_cache.mkdir(parents=True, exist_ok=True)
            ds = PersistentDataset(d, cls.tform, persistent_cache)
        else:
            ds = Dataset(d, cls.tform)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

        return loader

    @classmethod
    def load_seg_data(cls, path):
        path = Path(path)

        random.seed(0)
        images = [str(p.absolute()) for p in path.glob("*.nii")]
        d = [{"image": im} for im in images]

        # ds = CacheDataset(d, xform)
        ds = Dataset(d, cls.tform)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

        return loader

    @classmethod
    def load_model(cls, path):
        net_params = torch.load(path, map_location=cls.device)

        if isinstance(net_params, dict):
            cls.model.load_state_dict(net_params['net'])
        elif isinstance(net_params, torch.nn.Module):
            cls.model = net_params
        else:
            logging.error('Could not load network')
            return

        return cls.model

    @classmethod
    def sample_spline(cls, points):
        n = cls.spline_length
        length, channels = points.shape[1], 3
        t = torch.linspace(0, n - 1, length).floor().long().to(points.device)

        coeffs = natural_cubic_spline_coeffs(t.float(), points)

        spline = NaturalCubicSpline(coeffs)

        t1 = torch.linspace(0, n - 1, n).to(points.device)

        p2 = spline.evaluate(t1)
        return p2

    @classmethod
    def spline_loss(cls, pred, p1):
        # n = p1.shape[1]
        # length, channels = pred.shape[1], 3
        # t = torch.linspace(0, n - 1, length).floor().long().to(p1.device)
        #
        # coeffs = natural_cubic_spline_coeffs(t.float(), pred)
        #
        # spline = NaturalCubicSpline(coeffs)
        #
        # t1 = torch.linspace(0, n - 1, n).to(p1.device)

        if isinstance(pred, monai.data.MetaTensor):
            pred = pred.as_tensor()

        if isinstance(p1, monai.data.MetaTensor):
            p1 = p1.as_tensor()

        p2 = cls.sample_spline(pred)
        d = torch.cdist(p1.double(), p2.double())
        d_min1 = d.min(axis=2)[0]
        d_min2 = d.min(axis=1)[0]

        # Average symmetric loss
        d_min = torch.cat([d_min1, d_min2], dim=1)

        return (d_min.abs()).mean()

    @classmethod
    def convert_normalized_coord_to_mm(cls, points, meta_dict):
        extent = torch.nn.functional.pad(meta_dict['spatial_shape'] - 1, (0, 1), "constant", 1.0).to(points.device)
        orig_affine = meta_dict['original_affine'].to(points.device)

        lb = torch.matmul(orig_affine, torch.tensor([0,0,0,1], dtype=torch.float64, device=points.device))
        ub = torch.matmul(orig_affine, extent.double())

        slopes = 1 / (ub - lb)
        out = (points.double() + 0.5) / slopes[0:3]

        return out


    @classmethod
    def output_tform(cls, x):
        pred = []
        label = []
        for item in x:
            pred.append(cls.convert_normalized_coord_to_mm(item['pred'], item['image'].meta))
            label.append(cls.convert_normalized_coord_to_mm(item['label'], item['image'].meta))

        # xt = list_data_collate(x)
        return cls.spline_loss(list_data_collate(pred), list_data_collate(label))

    @classmethod
    def train(cls, data=None, use_val=False, load_checkpoint=None):
        config.print_config()

        if not data:
            dataPath = "U:/Documents/AnnulusTracking/"
        else:
            dataPath = data

        loader = cls.load_train_data(dataPath, use_val=use_val)

        net = cls.model


        # loss = DiceCELoss(softmax=True, include_background=False, lambda_dice=0.5)

        opt = Novograd(net.parameters(), 1e-2)

        # trainer = create_supervised_trainer(net, opt, loss, device, False, )
        trainer = SupervisedTrainer(
            device=cls.device,
            max_epochs=2000,
            train_data_loader=loader,
            network=net,
            optimizer=opt,
            loss_function=cls.spline_loss,
            # postprocessing=cls.output_tform,
            # decollate=False,
            key_train_metric={"train_MAE (mm)": Average(output_transform=cls.output_tform)},
            metric_cmp_fn=lambda curr, prev: curr < prev if prev > 0 else True,
            amp=True
        )

        # Load checkpoint if defined
        if load_checkpoint:
            checkpoint = torch.load(load_checkpoint)
            if checkpoint['trainer']:
                for k in checkpoint['trainer']:
                    trainer.state.__dict__[k] = checkpoint['trainer'][k]
                trainer.state.epoch = trainer.state.iteration // trainer.state.epoch_length
            checkpoint_loader = CheckpointLoader(load_checkpoint, {'net': net, 'opt': opt})
            checkpoint_loader.attach(trainer)

            logdir = Path(load_checkpoint).parent

        else:
            logdir = Path('./runs/')
            logdir.mkdir(exist_ok=True)
            dirs = sorted([int(x.name) for x in logdir.iterdir() if x.is_dir()])
            if not dirs:
                logdir = logdir.joinpath('0')
            else:
                logdir = logdir.joinpath(str(int(dirs[-1]) + 1))

        # Adaptive learning rate
        lr_scheduler = StepLR(opt, 1000)
        lr_handler = LrScheduleHandler(lr_scheduler)
        lr_handler.attach(trainer)

        ### optional section for checkpoint and tensorboard logging
        # adding checkpoint handler to save models (network params and optimizer stats) during training
        checkpoint_handler = CheckpointSaver(logdir, {'net': net, 'opt': opt, 'trainer': trainer}, n_saved=10, save_final=True,
                                             epoch_level=True, save_interval=200)
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
            loader = cls.load_val_data("U:/Documents/AnnulusTracking/", test=use_test)
        else:
            loader = cls.load_val_data(data, False, test=use_test)

        net = cls.load_model(load_checkpoint)

        logdir = Path(load_checkpoint).parent.joinpath('out')

        start = timer()
        avg = Average(device=cls.device)
        net.eval()
        with torch.no_grad():
            for batch in loader:
                im = batch['image'].to(cls.device)
                p1 = batch['label'].to(cls.device)

                p2 = net(im)
                batch['pred'] = p2.cpu()
                mm_mae = cls.output_tform(decollate_batch(batch))
                norm_mae = cls.spline_loss(p2, p1)
                avg.update(mm_mae)
                print("error (mm): {}".format(mm_mae), "error (norm): {}".format(norm_mae))

                out = decollate_batch(cls.sample_spline(p2))
                meta_dict = decollate_batch(im.meta)
                for o, m in zip(out, meta_dict):
                    test = cls.convert_normalized_coord_to_mm(p1.squeeze(),m)
                    f = logdir.joinpath(Path(m['filename_or_obj']).stem + '_pred.csv')
                    np.savetxt(str(f), o.detach().cpu().numpy())

        mean_mae = avg.compute()
        end = timer()

        print("Metric Mean_MAE: {}".format(mean_mae.item()))
        print("Elapsed time {}s".format(end-start))

    @classmethod
    def segment(cls, load_checkpoint, data):
        config.print_config()

        loader = cls.load_seg_data(data)

        device = cls.device
        net = cls.load_model(load_checkpoint)

        logdir = Path(data).joinpath('out')

        pred = AsDiscrete(argmax=True)

        saver = SaveImage(
            output_dir=str(logdir),
            output_postfix="seg",
            output_ext=".nii.gz",
            output_dtype=np.uint8,
        )

        net.eval()
        with torch.no_grad():
            for batch in loader:
                out = sliding_window_inference(batch['image'].to(device), (96,96,96), 16, net)
                out = cls.post_tform(decollate_batch({'pred': out}))
                meta_dict = decollate_batch(batch["image_meta_dict"])
                for o, m in zip(out, meta_dict):
                    saver(pred(o['pred']), m)



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

    seg_parse = subparsers.add_parser('segment', help='Segment images')
    seg_parse.add_argument('load', type=str, help='load from a given checkpoint')
    seg_parse.add_argument('data', type=str, help='data folder')

    return parser.parse_args()

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    start = timer()
    args = parse_args()
    if args.mode == 'validate':
        DEAD.validate(args.load, args.data, args.use_test)
    elif args.mode == 'train':
        DEAD.train(args.data, args.use_val, args.load)
    elif args.mode == 'segment':
        DEAD.segment(args.load, args.data)
    end = timer()
    print({"Total runtime: {}".format(end - start)})


if __name__ == "__main__":
    main()