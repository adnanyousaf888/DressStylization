import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2  # OpenCV for image processing

# Define class mappings
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
label_to_class = {v: k for k, v in class_mapping.items()}

# Function to load and configure the model
def load_model(model_path, num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes)

    # Load the trained weights and set the model to evaluation mode
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Image preparation function
def prepare_image(image):
    transform_fn = T.Compose([T.ToTensor()])
    img_tensor = transform_fn(image)
    return img_tensor, image

# Function to extract the dress mask from the image using the trained model
def get_dress_mask(model, image, confidence_threshold=0.3):
    img_tensor, img_pil = prepare_image(image)
    img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        predictions = model(img_tensor)

    pred_scores = predictions[0]['scores'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()
    pred_masks = predictions[0]['masks'].cpu().numpy()

    # Filter predictions by confidence threshold
    keep = pred_scores >= confidence_threshold
    pred_labels = pred_labels[keep]
    pred_masks = pred_masks[keep]
    pred_scores = pred_scores[keep]

    if len(pred_labels) == 0:
        print("No clothing detected in the image.")
        return None, None

    # Get the label with the highest confidence score
    best_index = pred_scores.argmax()
    best_label = pred_labels[best_index]
    best_mask = pred_masks[best_index]

    # Convert the mask to binary
    combined_mask = (best_mask[0] > 0.5).astype(np.uint8) * 255

    return combined_mask, np.array(img_pil)

# Function to overlay fabric on dress
def overlay_fabric_on_dress(image, mask, fabric):
    mask = mask.astype(np.uint8)
    mask[mask > 0] = 1

    fabric_resized = cv2.resize(fabric, (image.shape[1], image.shape[0]))

    # Neutralize the shirt area before applying the pattern
    image_copy = image.copy()
    image_copy[mask == 1] = [255, 255, 255]  # Set shirt area to white

    # Directly replace the mask area with fabric pattern
    result = image_copy.copy()
    for c in range(3):  # For each color channel
        result[:, :, c] = np.where(mask == 1, fabric_resized[:, :, c], image_copy[:, :, c])
    return result

# Blending function to preserve natural shadows and highlights
def blend_images(original_image, overlaid_image, mask, alpha=0.7):
    mask = mask.astype(np.float32)
    mask = cv2.GaussianBlur(mask, (15, 15), 0) / 255.0

    mask_expanded = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # Preserve the natural shading from the original dress in the blending
    blended = original_image * (1 - mask_expanded * alpha) + overlaid_image * (mask_expanded * alpha)
    blended = blended.astype(np.uint8)
    return blended

# Example usage
def process_images(model_path, content_image_path, design_patch_path, confidence_threshold=0.3, alpha=0.7):
    # Load model
    num_classes = len(class_mapping) + 1
    model = load_model(model_path, num_classes)

    # Load images
    content_image = Image.open(content_image_path).convert("RGB")
    design_patch = Image.open(design_patch_path).convert("RGB")

    # Process content image to get mask
    mask, image_np = get_dress_mask(model, content_image, confidence_threshold)
    if mask is None:
        return None

    # Convert images to NumPy arrays
    fabric = cv2.cvtColor(np.array(design_patch), cv2.COLOR_RGB2BGR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Overlay fabric on dress
    overlaid_image = overlay_fabric_on_dress(image_np, mask, fabric)

    # Blend images
    final_result = blend_images(image_np, overlaid_image, mask, alpha)
    return final_result

# Example usage call
final_image = process_images("mask_rcnn_IS_50.pth", "X:\\Sample convention images\\Dresses\\straight_trouser.png", "X:\\Sample convention images\\Dresses\\t4.jpg")
if final_image is not None:
    cv2.imwrite("D_stylized_dress.jpg", final_image)
