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
from monai.data import list_data_collate, decollate_batch,  PersistentDataset, Dataset

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
    SpatialPadd,
    SaveImaged,
)
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.utils.enums import CommonKeys as Keys

from utils import AimIgniteImageHandler

from generative_phantom.configs.model_config import (
    ModelConfig,
    InferenceConfig,    
    transforms_to_str,
)
import sys
import numpy as np
from aim.pytorch_ignite import AimLogger
from params import EXPERIMENTS
import pandas as pd

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

CODE_TESTING = False

def main(experiment_name, epoch):
    # Load the experiment configuration
    experiment = EXPERIMENTS[experiment_name]
    logging.info(f"Running experiment: {experiment_name}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if epoch == -1:
        # Load the last epoch
        checkpoints = glob(f" /yb107/math465/results/{experiment_name}/checkpoints/*")
        epoch = max([int(c.split("_")[-1].split(".")[0]) for c in checkpoints])
    
    config = InferenceConfig(
        name=experiment_name,
        description="Inference on Test Data",
        model_config_path=f" /yb107/math465/results/{experiment_name}/config.json",
        data_config_path=" /yb107/Datasets/AbdomenAtlas",
        model_checkpoint_path=f" /yb107/math465/results/{experiment_name}/checkpoints/{experiment_name}_checkpoint_{epoch}.pt",
        save_path=f" /yb107/math465/results/{experiment_name}/inferences/Test_{epoch}",
    )
    
    # need batch size * num_of_patches 
    model_config = ModelConfig(
        name=experiment_name,
        description="Using Custom UNetWithPrior Class, without Prior",
        save_path=f" /yb107/math465/results/{experiment_name}",
        preprocessed_config_path=" /yb107/Datasets/AbdomenAtlas",
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
    
    if not os.path.exists(model_config.save_path):
        os.makedirs(model_config.save_path)
    
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    base_path = model_config.preprocessed_config_path
    
    test_df = pd.read_csv("/home/yb107/ /math465_final/data/test.csv")
    
    test_images = test_df["name"].values
    # Shuffle the images
    np.random.shuffle(test_images)
                    
    if CODE_TESTING:        
        test_images = test_images[:5]

    test_files = [
        {Keys.IMAGE: f"{base_path}/{f}/ct.nii.gz", Keys.LABEL: f"{base_path}/{f}/segmentations/liver.nii.gz"}
        for f in test_images
    ]

    transforms = Compose(
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
            # CropForegroundd(keys=[Keys.IMAGE, Keys.LABEL], source_key=Keys.IMAGE),
            Orientationd(keys=[Keys.IMAGE, Keys.LABEL], axcodes="LPI"),
            Spacingd(
                keys=[Keys.IMAGE, Keys.LABEL], pixdim=(1.5, 1.5, 1.5), mode=("nearest")
            ),
            SpatialPadd(keys=[Keys.IMAGE, Keys.LABEL], spatial_size=model_config.patch_size),
        ]
    )

    ds = Dataset(
        data=test_files,
        transform=transforms,
        # cache_dir=" /yb107/math465/cache",
    )
    loader = DataLoader(
        ds,
        batch_size=1,
        num_workers=model_config.num_workers,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    # device = torch.device("cuda:0")
    
    UNet_metadata = {
        "spatial_dims": model_config.additional_params["UNet_metadata"]["spatial_dims"],
        "in_channels": model_config.additional_params["UNet_metadata"]["in_channels"],
        "out_channels": model_config.additional_params["UNet_metadata"]["out_channels"],
        "strides": model_config.additional_params["UNet_metadata"]["strides"],
        "norm": Norm.BATCH,
        
        "channels": model_config.additional_params["UNet_metadata"]["channels"],
    }
    
    if model_config.additional_params["prior_provided"]:
        if model_config.additional_params["prior_type"] == "all":
            logging.info("Using all data for prior from /home/yb107/ /math465_final/data/avg_prob_maps/avg_liver.npy")
            prob_map = np.load("/home/yb107/ /math465_final/data/avg_prob_maps/avg_liver.npy")
        else:          
            logging.info(f"Using {model_config.additional_params['training_data_size']} data for prior from /home/yb107/ /math465_final/data/avg_prob_maps/avg_liver_{model_config.additional_params['training_data_size']}.npy")
            prob_map = np.load(f"/home/yb107/ /math465_final/data/train_prob_maps/train_liver_{model_config.additional_params["training_data_size"]}.npy")
        
        UNet_metadata["prob_map"] = torch.Tensor(prob_map).unsqueeze(0).unsqueeze(0).to(device)
        
        if model_config.model == "UNETL":
            UNet_metadata["prob_map_latent_channels"] = 8
            UNet_metadata["prob_map_encoder"] = nn.Sequential(
                Convolution(spatial_dims=3, in_channels=1,  out_channels=8, kernel_size=3, strides=1, act=Act.PRELU, norm=Norm.BATCH),
                Convolution(spatial_dims=3, in_channels=8, out_channels=16, kernel_size=3, strides=2, act=Act.PRELU, norm=Norm.BATCH),
                Convolution(spatial_dims=3, in_channels=16, out_channels=8, kernel_size=3, strides=1, act=Act.PRELU, norm=Norm.BATCH),
            ).to(device)
        elif model_config.model == "UNET":
            UNet_metadata["prob_map_latent_channels"] = 8
            UNet_metadata["prob_map_encoder"] = nn.Sequential(
                Convolution(spatial_dims=3, in_channels=1,  out_channels=64, kernel_size=3, strides=2, act=Act.PRELU, norm=Norm.BATCH),
                Convolution(spatial_dims=3, in_channels=64, out_channels=32, kernel_size=3, strides=2, act=Act.PRELU, norm=Norm.BATCH),
                Convolution(spatial_dims=3, in_channels=32, out_channels=16, kernel_size=3, strides=2, act=Act.PRELU, norm=Norm.BATCH),
                Convolution(spatial_dims=3, in_channels=16, out_channels=8, kernel_size=3, strides=2, act=Act.PRELU, norm=Norm.BATCH),
            ).to(device)
        else:
            raise ValueError("Invalid model type")   
    
    if model_config.model == "UNET":
        model = UNetWithPrior(**UNet_metadata).to(device)
    elif model_config.model == "UNETL":
        model = UNetWithLastLayerPrior(**UNet_metadata).to(device)
    else:
        raise ValueError("Invalid model type")   
    model.load_state_dict(
        torch.load(config.model_checkpoint_path, map_location=device)["network"]
    )

    model_config.model = model.__class__.__name__
    model_config.pytorch_version = str(torch.__version__)
    model_config.monai_version = str(monai.__version__)

    if not CODE_TESTING:
        # model_config.save(model_config.save_path)
        config.save(config.save_path)
        

    # Ignite trainer expects batch=(img, seg) and returns output=loss at every iteration,
    # user can add output_transform to return other values, liksse: y_pred, y, etc.
    def prepare_batch(batch, device=None, non_blocking=True):
        return _prepare_batch(
            (batch[Keys.IMAGE], batch[Keys.LABEL]), device, non_blocking
        )

    metrics = {
        "Mean Dice": MeanDice(
            output_transform=from_engine([Keys.PRED, Keys.LABEL]),
            include_background=False,
        ),
    }
    
    def name_formatter(metadict: dict, saver: monai.transforms.Transform) -> dict:
        """Returns a kwargs dict for :py:meth:`FolderLayout.filename`,
        according to the input metadata and SaveImage transform."""
        subject = metadict["filename_or_obj"].split("/")[-2]
        patch_index = metadict.get(monai.utils.ImageMetaKey.PATCH_INDEX, None) if metadict else None
        return {"subject": f"{subject}", "idx": patch_index}


    post_pred = Compose(
        [
            AsDiscreted(keys=Keys.LABEL, to_onehot=model_config.output_channel),
            AsDiscreted(keys=Keys.PRED, argmax=True, to_onehot=model_config.output_channel),
            SaveImaged(keys= Keys.PRED, output_dir=f"{config.save_path}/output", separate_folder=False, output_name_formatter=name_formatter, output_postfix="seg"),
        ]
    )
    inferer = SlidingWindowInferer(
        roi_size=model_config.patch_size,
        sw_batch_size=model_config.batch_size,
        overlap=0.8,
    )
    
    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=loader,
        network=model,
        prepare_batch=prepare_batch,
        key_val_metric=metrics,
        postprocessing=post_pred,
        non_blocking=True,
        inferer=inferer,
    )

    # Create a logger
    if CODE_TESTING:
        aim_logger = AimLogger(
            repo=model_config.aim_repo_dir,
            experiment="Code Testing",
        )
        aim_logger.experiment.add_tag("code_testing")
    else:
        aim_logger = AimLogger(
            repo=model_config.aim_repo_dir,
            experiment=config.name,
        )
        aim_logger.experiment.add_tag("Testing")
        aim_logger.experiment.description = config.description

    aim_logger.experiment.log_warning(
        f" Number of Test images {len(loader)}"
    )
    
    aim_logger.log_params(config.__dict__)
    # aim_logger.experiment["UNet_metadata"] = UNet_metadata

    aim_logger.attach_output_handler(
        evaluator,
        event_name=Events.EPOCH_COMPLETED,
        tag="val",
        metric_names=["Mean Dice"]
    )
    
    evaluator.add_event_handler(
        Events.ITERATION_COMPLETED,
        lambda engine: logging.info(f"Validation Iteration {engine.state.iteration}"),
    )
    
    aim_logger.attach(
        evaluator,
        AimIgniteImageHandler(
            "ImagePredictionLabel",
            output_transform=from_engine([Keys.IMAGE, Keys.LABEL,Keys.PRED], first=True),
        ),
        event_name=Events.ITERATION_COMPLETED,
    )

    evaluator.run()

if __name__ == "__main__":
    # Accepts a single argument, the experiment name
    
    # print(sys.argv)
    if len(sys.argv) > 1:
        experiment_name = sys.argv[1]
        main(experiment_name)
    else:
        
        
        main("UNETL_True_256_1000_training", -1)
        # main("UNETL_True_128_128_training", -1)
        # main("UNETL_True_128_128_all", -1)
        # main("UNETL_True_32_64_all", -1)
        # main("UNETL_True_64_128_all", -1)
        # main("UNETL_True_32_128_all", -1)
        # main("UNETL_True_64_256_training", 72)
        # main("UNETL_True_256_64_all", -1)
        # ----
        # main("UNET_True_32_128_all", -1)
        # main("UNET_True_32_64_training", -1)

        # main("UNET_False_256_1000_training", 188)
        # main("UNET_False_64_256_all", -1)
        # main("UNET_True_64_256_all", -1)
        # main("UNET_False_64_128_all", -1)
        # main("UNET_True_64_128_all", 83)
        # main("UNET_False_32_64_all", -1)
        # main("UNET_True_32_64_all", 136)
        # main("UNET_True_64_256_all", -1)
        # main("UNET_False_32_128_training", 53)
        # main("UNET_True_32_128_training", 117)
        # main("UNET_False_128_128_training", 160)
        # main("UNET_True_128_128_all", 136)

# name = f"UNET_{prior_provided}_{channels[-1]}_{training_data_size}_{prior_type}"
    
