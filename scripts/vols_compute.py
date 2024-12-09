import numpy as np
import nibabel as nib
import pandas as pd
import os
from glob import glob
base_dir = " /yb107/Datasets/AbdomenAtlas/"
liver_imgs = sorted(glob(base_dir + "*/segmentations/liver.nii.gz"))
print(f"Total {len(liver_imgs)}")

vol_df = pd.DataFrame(columns=["name", "volume"])

for idx, i in enumerate(liver_imgs):
    img = nib.load(i)
    data = img.get_fdata()
    spacing = img.header.get_zooms()
    vol = np.sum(data) * np.prod(spacing) / 1000
    vol_df = pd.concat([vol_df, pd.DataFrame({"name":i.split("/")[-3], "volume": vol}, index=[0])], ignore_index=True)
    
    if idx % 100 == 0:
        print(f"Processed {idx} images")
        vol_df.to_csv("/home/yb107/ /math465_final/data/volumes.csv", index=False)
        
vol_df.to_csv("/home/yb107/ /math465_final/data/volumes.csv", index=False)