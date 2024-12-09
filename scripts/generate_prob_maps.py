import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from  monai.transforms import LoadImage, Compose, CropForeground, ResizeWithPadOrCrop, EnsureChannelFirst, Spacing, Orientation
from generative_phantom.monai_helpers import get_big_fig
import os
from glob import glob
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)

base_dir = "/dataset/public_dataset/CT_images/abdomen/AbdomenAtlasMini/AbdomenAtlas1.0/"
# dirs = [i for i in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, i))]
# liver_imgs = glob(base_dir + "*/segmentations/liver.nii.gz")
# liver_imgs = liver_imgs[:2500]

df = pd.read_csv("/home/yb107/math465_final/data/test.csv")
avoid = df["name"].values
all_names =  pd.read_csv("/home/yb107/ /math465_final/data/volumes3000.csv")["name"].values
liver_imgs = [os.path.join(base_dir, i, "segmentations", "liver.nii.gz" ) for i in all_names if i not in avoid]
# liver_imgs = [os.path.join(base_dir, i, "segmentations", "liver.nii.gz" ) for i in df["name"].values]


transforms = Compose([LoadImage(), EnsureChannelFirst(), Spacing((3,3,3)), Orientation("LPI"), CropForeground(), ResizeWithPadOrCrop((128, 128, 128))])

avg_liver = np.zeros((128, 128, 128))
logging.info(f"Total {len(liver_imgs)} images")
for idx, img in enumerate(liver_imgs):
    trans_img = transforms(img).squeeze()
    avg_liver = avg_liver + trans_img
    
    if idx % 300 == 0 and idx != 0:
        logging.info(f"Processed {idx} images")
        np.save(f"/home/yb107/ /math465_final/data/avg_prob_maps/avg_liver_{idx}.npy", avg_liver / idx)    
        
np.save(f"/home/yb107/ /math465_final/data/avg_prob_maps/avg_liver.npy", avg_liver / len(liver_imgs))