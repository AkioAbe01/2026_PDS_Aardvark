import pandas as pd
from skimage import io
import os


def load_metadata():
    """Loads the metadata"""
    data = pd.read_csv("data/metadata.csv")
    return data

def build_image_path(row):
    """ Builds a correct path for the image"""

    image_id = row["img_id"]
    path = "data/imgs/" + image_id
    return path

def build_mask_path(row):
    """builds a correct path for the mask"""

    image_id = row["img_id"]
    path = "data/masks/" + image_id[:-4] + "_mask.png" 
    return path

   
def filter_missing_masks(metadata):
    """ Creates a new df that will not include images that have no masks and masks, that are not applied to any image in the metadata"""

    mask_files = set(os.listdir("data/masks")) #set of masks
    mask = metadata.apply(lambda row: (row["img_id"][:-4] + "_mask.png") in mask_files, axis=1) #check if expected mask is in reality in the given data
    filtered_metadata = metadata[mask]

    return filtered_metadata

#create a new df that will not include images and masks that have more than one lesion
def filter_multiple_lesions(path):
    #to be implemented
    return

def load_image(path):
    """ Loads image"""

    image = io.imread(path)
    return image

def load_mask(path):
    """Loads a mask"""

    mask = io.imread(path)
    return mask


def get_image_mask_pairs(metadata):
    """ Gets the mask-image pair and ensures they come from the same lesion"""
    
    pairs = []
    for _, row in metadata.iterrows():
        image_path = build_image_path(row)
        mask_path = build_mask_path(row)
        pairs.append((image_path, mask_path))
    return pairs


def df_binary_cancer_label(metadata):
    """
    Create a feature table with img_id and binary cancer label.

    1 = cancerous
    0 = not cancerous
    """
    cancerous_types = ["BCC", "MEL", "SCC"]

    feature = metadata.copy()
    feature["Cancerous"] = feature["diagnostic"].isin(cancerous_types).astype(int)

    feature = feature[["img_id", "Cancerous"]]

    return feature