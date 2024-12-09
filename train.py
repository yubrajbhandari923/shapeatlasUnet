import logging
import sys, os
from glob import glob

import monai.data
import monai.transforms
import torch
import torch.nn as nn

from ignite.engine import (
    Events,
    _prepare_batch,
)
from ignite.handlers import EarlyStopping, ModelCheckpoint, global_step_from_engine
from torch.utils.data import DataLoader

import monai
from monai.data import list_data_collate, decollate_batch,  PersistentDataset
# from monai.networks.nets import UNet
from model import UNetWithPrior, UNetWithLastLayerPrior
from monai.networks.layers.factories import Norm, Act
from monai.networks.blocks import Convolution
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.handlers import (
    MeanDice,
    StatsHandler,
    IgniteMetricHandler,
    stopping_fn_from_metric,
    from_engine,
)
from monai.transforms import (
    EnsureChannelFirstd,
    AsDiscreted,
    Compose,
    LoadImaged,
    CropForegroundd,
    SaveImage,
    Spacingd,
    Orientationd,
    ResizeWithPadOrCropd,
    CropForegroundd,
    Flipd,
    ScaleIntensityRanged,
    RandCropByPosNegLabeld,
    SpatialPadd
)
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.utils.enums import CommonKeys as Keys

from utils import AimIgniteImageHandler

from generative_phantom.configs.model_config import (
    ModelConfig,
    PreprocessedDataConfig,
    transforms_to_str,
)
import sys
import numpy as np
from aim.pytorch_ignite import AimLogger
from params import EXPERIMENTS
import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

CODE_TESTING = False


def main(experiment_name):
    # Load the experiment configuration
    experiment = EXPERIMENTS[experiment_name]
    logging.info(f"Running experiment: {experiment_name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # need batch size * num_of_patches 
    config = ModelConfig(
        name=experiment_name,
        description="Using Custom UNetWithPrior Class,",
        save_path=f" /yb107/math465/results/{experiment_name}",
        preprocessed_config_path="/yb107/Datasets/AbdomenAtlas",
        epochs=500,
        batch_size=8,
        num_workers=12,
        device="cuda",
        output_channel=1 + 1,  # Fix
        loss="DiceLoss",
        optimizer="Adam",
        patch_size=(128, 128, 128),
        patch_overlap=0,
        lr=0.001,
        beta_1=0.3,
        beta_2=0.999,
        early_stopping=True,
        early_stopping_patience=30,
        log_interval=1,
        validation_interval=1,
        visualization_interval=1,
        aim_repo_dir=" /yb107/math465/results/aim",
        additional_params={
            "map": {
                "background": 0,
                "liver": 1,
            },
            "UNet_metadata": {
                "spatial_dims": 3,
                "in_channels": 1,
                "out_channels": 2,
                "strides": (2, 2, 2, 2),
                "norm": "batch",
                
                "channels": experiment["channels"],
            },
            "prior_provided": experiment["prior_provided"],
            "training_data_size": experiment["training_data_size"],
            "prior_type": experiment["prior_type"],  
        },
        model=experiment["model"],
    )
    monai.config.print_config()
    
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    base_path = config.preprocessed_config_path
    
    train_df = pd.read_csv("/home/yb107/ /math465_final/data/train.csv")
    val_df = pd.read_csv("/home/yb107/ /math465_final/data/val.csv")
    
    train_images = train_df["name"].values
    val_images = val_df["name"].values

    if config.additional_params["training_data_size"] != 1000:    
        train_images = train_images[:config.additional_params["training_data_size"]]
        val_images = val_images[:config.additional_params["training_data_size"]]
                    
    if CODE_TESTING:        
        train_images = train_images[:20]
        val_images = val_images[:4]

    train_files = [
        {Keys.IMAGE: f"{base_path}/{f}/ct.nii.gz", Keys.LABEL: f"{base_path}/{f}/segmentations/liver.nii.gz"}
        for f in train_images
    ]

    val_files = [
        {Keys.IMAGE: f"{base_path}/{f}/ct.nii.gz", Keys.LABEL: f"{base_path}/{f}/segmentations/liver.nii.gz"}
        for f in val_images
    ]

    train_transforms = Compose(
        [
            LoadImaged(keys=[Keys.IMAGE, Keys.LABEL]),
            EnsureChannelFirstd(keys=[Keys.IMAGE, Keys.LABEL]),
            ScaleIntensityRanged(
                keys=[Keys.IMAGE],
                a_min=-1000,
                a_max=1000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=[Keys.IMAGE, Keys.LABEL], source_key=Keys.IMAGE),
            Orientationd(keys=[Keys.IMAGE, Keys.LABEL], axcodes="LPI"),
            Spacingd(
                keys=[Keys.IMAGE, Keys.LABEL], pixdim=(1.5, 1.5, 1.5), mode=("nearest")
            ),
            SpatialPadd(keys=[Keys.IMAGE, Keys.LABEL], spatial_size=config.patch_size), 
            RandCropByPosNegLabeld(
                keys=[Keys.IMAGE, Keys.LABEL],
                label_key=Keys.LABEL,
                spatial_size=config.patch_size,
                pos=1,
                neg=1,
                num_samples=1,
                allow_smaller=True,
            ),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=[Keys.IMAGE, Keys.LABEL]),
            EnsureChannelFirstd(keys=[Keys.IMAGE, Keys.LABEL]),
            ScaleIntensityRanged(
                keys=[Keys.IMAGE],
                a_min=-1000,
                a_max=1000,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=[Keys.IMAGE, Keys.LABEL], source_key=Keys.IMAGE),
            Orientationd(keys=[Keys.IMAGE, Keys.LABEL], axcodes="LPI"),
            Spacingd(
                keys=[Keys.IMAGE, Keys.LABEL], pixdim=(1.5, 1.5, 1.5), mode=("nearest")
            ),
            SpatialPadd(keys=[Keys.IMAGE, Keys.LABEL], spatial_size=config.patch_size),
        ]
    )
    # create a training data loader
    train_ds = PersistentDataset(
        data=train_files,
        transform=train_transforms,
        cache_dir=" /yb107/math465/cache/",
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )

    # create a validation data loader
    val_ds = PersistentDataset(
        data=val_files,
        transform=val_transforms,
        cache_dir=" /yb107/math465/cache",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=config.num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    device = torch.device("cuda:0")
    
    UNet_metadata = {
        "spatial_dims": config.additional_params["UNet_metadata"]["spatial_dims"],
        "in_channels": config.additional_params["UNet_metadata"]["in_channels"],
        "out_channels": config.additional_params["UNet_metadata"]["out_channels"],
        "strides": config.additional_params["UNet_metadata"]["strides"],
        "norm": Norm.BATCH,
        
        "channels": config.additional_params["UNet_metadata"]["channels"],
    }
    
    if config.additional_params["prior_provided"]:
        if config.additional_params["prior_type"] == "all":
            logging.info("Using all data for prior from /home/yb107/ /math465_final/data/avg_prob_maps/avg_liver.npy")
            prob_map = np.load("/home/yb107/ /math465_final/data/avg_prob_maps/avg_liver.npy")
        else:          
            logging.info(f"Using {config.additional_params['training_data_size']} data for prior from /home/yb107/ /math465_final/data/avg_prob_maps/avg_liver_{config.additional_params['training_data_size']}.npy")
            prob_map = np.load(f"/home/yb107/ /math465_final/data/train_prob_maps/train_liver_{config.additional_params["training_data_size"]}.npy")
        
        UNet_metadata["prob_map"] = torch.Tensor(prob_map).unsqueeze(0).unsqueeze(0).to(device)
        
        if config.model == "UNETL":
            UNet_metadata["prob_map_latent_channels"] = 8
            UNet_metadata["prob_map_encoder"] = nn.Sequential(
                Convolution(spatial_dims=3, in_channels=1,  out_channels=8, kernel_size=3, strides=1, act=Act.PRELU, norm=Norm.BATCH),
                Convolution(spatial_dims=3, in_channels=8, out_channels=16, kernel_size=3, strides=2, act=Act.PRELU, norm=Norm.BATCH),
                Convolution(spatial_dims=3, in_channels=16, out_channels=8, kernel_size=3, strides=1, act=Act.PRELU, norm=Norm.BATCH),
            ).to(device)
        elif config.model == "UNET":
            UNet_metadata["prob_map_latent_channels"] = 8
            UNet_metadata["prob_map_encoder"] = nn.Sequential(
                Convolution(spatial_dims=3, in_channels=1,  out_channels=64, kernel_size=3, strides=2, act=Act.PRELU, norm=Norm.BATCH),
                Convolution(spatial_dims=3, in_channels=64, out_channels=32, kernel_size=3, strides=2, act=Act.PRELU, norm=Norm.BATCH),
                Convolution(spatial_dims=3, in_channels=32, out_channels=16, kernel_size=3, strides=2, act=Act.PRELU, norm=Norm.BATCH),
                Convolution(spatial_dims=3, in_channels=16, out_channels=8, kernel_size=3, strides=2, act=Act.PRELU, norm=Norm.BATCH),
            ).to(device)
        else:
            raise ValueError("Invalid model type")   
    
    if config.model == "UNET":
        model = UNetWithPrior(**UNet_metadata).to(device)
    elif config.model == "UNETL":
        model = UNetWithLastLayerPrior(**UNet_metadata).to(device)
    else:
        raise ValueError("Invalid model type")        

    # model = UNetWithPrior(**UNet_metadata).to(device)
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    # create UNet, DiceLoss and Adam optimizer
    config.model = model.__class__.__name__
    config.pytorch_version = str(torch.__version__)
    config.monai_version = str(monai.__version__)
    config.train_transforms = transforms_to_str(train_transforms)
    config.train_length = len(train_ds)
    config.val_length = len(val_ds)

    if not CODE_TESTING:
        config.save(config.save_path)

    # Ignite trainer expects batch=(img, seg) and returns output=loss at every iteration,
    # user can add output_transform to return other values, liksse: y_pred, y, etc.
    def prepare_batch(batch, device=None, non_blocking=True):
        return _prepare_batch(
            (batch[Keys.IMAGE], batch[Keys.LABEL]), device, non_blocking
        )

    # This part is ignite-specific
    trainer = SupervisedTrainer(
        device=device,
        max_epochs=config.epochs,
        train_data_loader=train_loader,
        network=model,
        optimizer=optimizer,
        loss_function=loss_function,
        prepare_batch=prepare_batch,
    )

    metrics = {
        "Mean Dice": MeanDice(
            output_transform=from_engine([Keys.PRED, Keys.LABEL]),
            include_background=False,
        ),
    }

    post_pred = Compose(
        [
            AsDiscreted(keys=Keys.LABEL, to_onehot=config.output_channel),
            AsDiscreted(keys=Keys.PRED, argmax=True, to_onehot=config.output_channel),
        ]
    )
    inferer = SlidingWindowInferer(
        roi_size=config.patch_size,
        sw_batch_size=3,
        overlap=0.25,
    )
    val_evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=model,
        prepare_batch=prepare_batch,
        key_val_metric=metrics,
        postprocessing=post_pred,
        non_blocking=True,
        inferer=inferer,
    )

    if not CODE_TESTING:
        # adding checkpoint handler to save models (network params and optimizer stats) during training
        checkpoint_handler = ModelCheckpoint(
            config.save_path + "/checkpoints",
            filename_prefix=config.name,
            n_saved=None,
            require_empty=False,
            global_step_transform=lambda eng, event: eng.state.epoch,
        )
        trainer.add_event_handler(
            event_name=Events.EPOCH_COMPLETED,
            handler=checkpoint_handler,
            to_save={"network": model, "optimizer": optimizer},
        )

    train_stats_handler = StatsHandler(
        name="trainer",
        output_transform=from_engine(["loss"], first=True),
        iteration_log=False,
    )
    train_stats_handler.attach(trainer)

    @trainer.on(Events.EPOCH_COMPLETED(every=config.validation_interval))
    def run_validation(engine):
        val_evaluator.run()

    # add early stopping handler to evaluator
    early_stopper = EarlyStopping(
        patience=config.early_stopping_patience,
        score_function=stopping_fn_from_metric("Mean Dice"),
        trainer=trainer,
    )

    val_evaluator.add_event_handler(
        event_name=Events.EPOCH_COMPLETED, handler=early_stopper
    )

    # Create a logger
    if CODE_TESTING:
        aim_logger = AimLogger(
            repo=config.aim_repo_dir,
            experiment="Code Testing",
        )
        aim_logger.experiment.add_tag("code_testing")
    else:
        aim_logger = AimLogger(
            repo=config.aim_repo_dir,
            experiment=config.name,
        )
        aim_logger.experiment.add_tag("Training")
        aim_logger.experiment.description = config.description

    aim_logger.experiment.log_warning(
        "WARNING: Number of Val images is not divisible by 4"
        if len(val_loader) % 4 != 0
        else "CHECKED: Number of Val images is divisible by 4"
    )
    aim_logger.log_params(config.__dict__)
    # aim_logger.experiment["UNet_metadata"] = UNet_metadata

    aim_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED,
        tag="Iteration Dice Loss",
        output_transform=from_engine(["loss"], first=True),
    )
    aim_logger.attach_output_handler(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        tag="Epoch Dice Loss",
        output_transform=from_engine(["loss"], first=True),
    )

    aim_logger.attach_output_handler(
        val_evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="val",
        metric_names=["Mean Dice"],
        global_step_transform=global_step_from_engine(trainer),
    )

    aim_logger.attach(
        val_evaluator,
        AimIgniteImageHandler(
            "PredictionLabel",
            output_transform=from_engine([Keys.PRED, Keys.LABEL], first=True),
            global_step_transform=global_step_from_engine(trainer),
        ),
        event_name=Events.ITERATION_COMPLETED(
            every=len(val_loader) // 4 if (len(val_loader) % 4) == 0 else len(val_loader) // 2
        ),
    )

    # am_logger.attach(
    #     val_evaluator,
    #     AimIgniteImageHandler(
    #         "Label",
    #         output_transform=from_engine([Keys.LABEL], first=True),
    #         global_step_transform=global_step_from_engine(trainer),
    #         plot_once=True,
    #         log_unique_values=False,
    #     ),
    #     event_name=Events.ITERATION_COMPLETED(
    #         every=2 if (len(val_loader) // 4) == 0 else len(val_loader) // 4
    #     ),
    # )

    aim_logger.attach(
        val_evaluator,
        AimIgniteImageHandler(
            "ImageLabel",
            output_transform=from_engine([Keys.IMAGE, Keys.LABEL], first=True),
            global_step_transform=global_step_from_engine(trainer),
            # plot_once=True,
        ),
        event_name=Events.ITERATION_COMPLETED(
            every=len(val_loader) // 4 if (len(val_loader) % 4) == 0 else len(val_loader) // 2
        ),
    )
    
    aim_logger.experiment.log_info(
        f"{len(train_loader)} training batches, {len(val_loader)} validation batches"
    )

    trainer.run()

if __name__ == "__main__":
    # Accepts a single argument, the experiment name
    
    # print(sys.argv)
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
        main(experiment_name)
    else:
        main("UNETL_True_256_1000_training")
        # main("UNET_True_32_64_training")
        

# name = f"UNET_{prior_provided}_{channels[-1]}_{training_data_size}_{prior_type}"
    
