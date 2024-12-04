import torch
import torchvision

import torchvision.transforms as T
from PIL import Image
import numpy as np

import cv2


# class mappings
class_mapping = {
    0: 'buttondown_shirt',
    1: 'capri',
    2: 'dhoti_shalwar',
    3: 'kurta',
    4: 'plazzo',
    5: 'shalwar',
    6: 'short_kurti',
    7: 'straight_shirt',
    8: 'trouser'
}


# labels to class conversion
label_to_class = {}
for key, value in class_mapping.items():
    label_to_class[value] = key


# Func for loading
# model from the directory
def load_model(model_path, num_classes):

    # base model > faster
    base_model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)

    # region of interest
    in_features = base_model.roi_heads.box_predictor.cls_score.in_features


    #box predictor changing
    base_model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    # model weights
    model_weights = torch.load(model_path, map_location=torch.device('cpu'))

    base_model.load_state_dict(model_weights)

    # evalution of model
    base_model.eval()
    return base_model

# prepararation of images
def prepare_image(image):

    #convertiong ot tensors
    transform_pipeline = T.Compose([T.ToTensor()])
    image_tensor = transform_pipeline(image)

    #returning tensor and actual images=
    return image_tensor, image


# finding mask of the region we want
def get_dress_mask(model, image, confidence_threshold=0.3):

    image_tensor, original_image = prepare_image(image)

    # unsqueezing the layer
    image_tensor_batch = image_tensor.unsqueeze(0)
     #no gradients
    with torch.no_grad():
        predictions = model(image_tensor_batch)
    
    # predicitons 
    # scores
    predicted_scores = predictions[0]['scores'].cpu().numpy()
    #labels from the images
    predicted_labels = predictions[0]['labels'].cpu().numpy()

    #generated masks >>> instance seg.
    predicted_masks = predictions[0]['masks'].cpu().numpy()

    # if pred score >= ... ignore
    valid_indices = predicted_scores >= confidence_threshold

    predicted_scores = predicted_scores[valid_indices]
    predicted_labels = predicted_labels[valid_indices]
    predicted_masks = predicted_masks[valid_indices]

    # in case there's no label predicted
    if len(predicted_labels) == 0:
        print("No clothing items detected in the image.")
        return None, None
    
    # at last layer
    max_confidence_index = predicted_scores.argmax()

    # picking best label predicted
    best_label = predicted_labels[max_confidence_index]
    # picking best mask predicted
    best_mask = predicted_masks[max_confidence_index]

    # generating binary mask and mul by 255
    binary_mask = (best_mask[0] > 0.5).astype(np.uint8) * 255
    return binary_mask, np.array(original_image)


# overlaying content to iamges
def overlay_fabric_on_dress(original_image, binary_mask, fabric_pattern):

    # as uint8
    binary_mask = binary_mask.astype(np.uint8)

    # if grearer 1
    binary_mask[binary_mask > 0] = 1

    # resized needed for further process
    fabric_resized = cv2.resize(fabric_pattern, (original_image.shape[1], original_image.shape[0]))

    # making deep copy of original for context
    image_with_mask_removed = original_image.copy()

    # checking if white back.
    image_with_mask_removed[binary_mask == 1] = [255, 255, 255]

    # deep copy for futher computation
    fabric_overlay_result = image_with_mask_removed.copy()
    # over channels 
    for channel in range(3):
        fabric_overlay_result[:, :, channel] = np.where(
            binary_mask == 1, fabric_resized[:, :, channel], image_with_mask_removed[:, :, channel]
        )
    return fabric_overlay_result


# blending d to c
def blend_images(original_image, overlaid_image, mask, alpha=0.7):
   
    # alpha is hyperParameter
    mask = mask.astype(np.float32)

    # Guassian distribution for blurrness
    # and divided by last elem of rgb
    blurred_mask = cv2.GaussianBlur(mask, (15, 15), 0) / 255.0
    mask_expanded = np.repeat(blurred_mask[:, :, np.newaxis], 3, axis=2)

    #blend operation
    blended_result = original_image * (1 - mask_expanded * alpha) + overlaid_image * (mask_expanded * alpha)
    blended_result = blended_result.astype(np.uint8)
    return blended_result

# making images ready
def process_images(model_path, content_image_path, design_patch_path, confidence_threshold=0.3, alpha=0.7):
    # confidence and alpha is hyperP
    # can adjust accordingly

    # adding 1 as back.
    num_classes = len(class_mapping) + 1
    # loading model and no, of classes
    model = load_model(model_path, num_classes)

    # Red green blue conversion for c
    content_image = Image.open(content_image_path).convert("RGB")

    # Red green blue conversion for d
    design_patch = Image.open(design_patch_path).convert("RGB")

    # extracting roi fron the input
    dress_mask, original_image_np = get_dress_mask(model, content_image, confidence_threshold)
    if dress_mask is None:
        return None
    
    #performing some CV operations
    fabric_pattern = cv2.cvtColor(np.array(design_patch), cv2.COLOR_RGB2BGR)
    original_image_np = cv2.cvtColor(original_image_np, cv2.COLOR_RGB2BGR)

    # calling funcs
    overlay_result = overlay_fabric_on_dress(original_image_np, dress_mask, fabric_pattern)
    #final output
    final_output = blend_images(original_image_np, overlay_result, dress_mask, alpha)
    return final_output

# output imges to save
output_image = process_images(
    # model path
    model_path="mask_rcnn_IS_50.pth",
    
    #content image path
    content_image_path="X:\\Sample convention images\\Dresses\\straight_trouser.png",

    # design image path
    design_patch_path="X:\\Sample convention images\\Dresses\\t4.jpg"
)

#final output
if output_image is not None:
    cv2.imwrite("_stylized_dress.jpg", output_image)
