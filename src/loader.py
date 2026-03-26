import pandas as pd
from skimage import io
import os

#function load the metadata
def load_metadata():
    data = pd.read_csv("data/metadata.csv")
    return data

#should be in main.py   0
metadata = load_metadata()

def build_image_path(row):
    image_id = row["img_id"]
    path = "data/imgs/" + image_id
    return path

def build_mask_path(row):
    image_id = row["img_id"] #eg: PAT_1516_1765_530.png
    path = "data/masks/" + image_id[:-4] + "_mask.png" 
    return path

#create a new df that will not include images that have no masks    
def filter_missing_masks(metadata):
    mask_files = set(os.listdir("data/masks")) #set of masks
    mask = metadata.apply(lambda row: (row["img_id"][:-4] + "_mask.png") in mask_files, axis=1) #check if expected mask is in reality in the given data
    filtered_metadata = metadata[mask]

    return filtered_metadata

#create a new df that will not include images and masks that have more than one lesion
def filter_multiple_lesions(path):
    #to be implemented
    return

#load image and mask
def load_image(path):
    image = io.imread(path)
    return image

def load_mask(path):
    mask = io.imread(path)
    return mask

#get the mask_image pair and ensure they come from the same lesion
def get_image_mask_pairs(metadata):
    pairs = []
    for _, row in metadata.iterrows():
        image_path = build_image_path(row)
        mask_path = build_mask_path(row)
        pairs.append((image_path, mask_path))
    return pairs

