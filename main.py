#------------------------was already created in the repo---------------------------
def main(features_path, prediction_results_path, model_path, load_model):
    """
    Docstring for main
    
    :param features_path: Path to the features csv used as input to the model (e.g. ./data/features.csv).
    :param prediction_results_path: Path to save the output predictions of the model (e.g. ./result/predictions/predictions_MODEL.csv).
    :param model_path: Path to save or load the trained model (e.g. ./result/predictions/predictions_MODEL.csv).
    :param load_model: Boolean to train the model and save it to model_path if False, load it from model_path if True. 
    """
    
    # load dataset CSV file

    # split the dataset into training and testing sets.

    if load_model:
        # load the model
        pass
    else:
        # train the classifier (using logistic regression as an example)

        # save the model.
        pass

    # test the classifier.


    # write test results to CSV.
#----------------------------------------------------------------------------------

from src.loader import (
    load_metadata,
    filter_missing_masks,
    get_image_mask_pairs,
    load_image,
    load_mask
)

print("--- Testing loader.py ---")

metadata = load_metadata()
print("Total metadata rows: ", len(metadata))

filtered_metadata = filter_missing_masks(metadata)
print("After filtering masks: ", len(filtered_metadata))

pairs = get_image_mask_pairs(filtered_metadata)
print("Number of pairs: ", len(pairs))


image_path, mask_path = pairs[0]
print(f"First image path: {image_path}\nFirst mask path: {mask_path}")

image = load_image(image_path)
mask = load_mask(mask_path)
print("Image shape:", image.shape)
print("Mask shape:", mask.shape)

print("---Done---")


#How to activate feature using main.py and loader.py (a little cheatsheet)
'''
Lets say we have featureA.py that requires a mask

from src.featureA import func_name

metadata = load_metadata()
filtered_metadata = filter_missing_masks(metadata)
pairs = get_image_mask_pairs(filtered_metadata)

feature_values = []

for image_path, mask_path in pairs:
    image = load_image(image_path)
    mask = load_mask(mask_path)

    value = func_name(mask)


    feature_values.append(value)

print(feature_values[:5])

'''


#------------------------was already created in the repo---------------------------
if __name__ == "__main__":
    features_path = "./data/features.csv"
    prediction_results_path = "./result/predictions/predictions_MODEL.csv"
    model_path = "./result/predictions/predictions_MODEL.csv"
    load_model = False

    main(features_path, prediction_results_path,model_path,load_model)
#----------------------------------------------------------------------------------