import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

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

# Function to load the TensorFlow model
def load_model(model_path):
    model = tf.saved_model.load(model_path)
    return model

# Image preparation function
def prepare_image(image):
    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    return input_tensor, image_np

# Function to extract the dress mask from the image using the trained model
def get_dress_mask(model, image, confidence_threshold=0.3):
    input_tensor, image_np = prepare_image(image)
    detections = model(input_tensor)

    # Extract predictions
    detection_scores = detections['detection_scores'].numpy()[0]
    detection_classes = detections['detection_classes'].numpy()[0].astype(np.int32)
    detection_masks = detections.get('detection_masks', None)
    
    if detection_masks is not None:
        detection_masks = detection_masks.numpy()[0]

    # Filter predictions by confidence threshold
    keep_indices = detection_scores >= confidence_threshold
    detection_classes = detection_classes[keep_indices]
    detection_scores = detection_scores[keep_indices]
    
    if detection_masks is not None:
        detection_masks = detection_masks[keep_indices]

    if len(detection_classes) == 0:
        print("No clothing detected in the image.")
        return None, None

    # Get the label with the highest confidence score
    best_index = np.argmax(detection_scores)
    best_class = detection_classes[best_index]
    best_mask = detection_masks[best_index] if detection_masks is not None else None

    # Convert the mask to binary
    combined_mask = (best_mask > 0.5).astype(np.uint8) * 255 if best_mask is not None else None
    return combined_mask, image_np

# Function to overlay fabric on dress
def overlay_fabric_on_dress(image, mask, fabric):
    mask = mask.astype(np.uint8)
    mask[mask > 0] = 1

    fabric_resized = cv2.resize(fabric, (image.shape[1], image.shape[0]))

    image_copy = image.copy()
    image_copy[mask == 1] = [255, 255, 255]

    result = image_copy.copy()
    for c in range(3):
        result[:, :, c] = np.where(mask == 1, fabric_resized[:, :, c], image_copy[:, :, c])
    return result

# Blending function to preserve natural shadows and highlights
def blend_images(original_image, overlaid_image, mask, alpha=0.7):
    mask = mask.astype(np.float32)
    mask = cv2.GaussianBlur(mask, (15, 15), 0) / 255.0

    mask_expanded = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    blended = original_image * (1 - mask_expanded * alpha) + overlaid_image * (mask_expanded * alpha)
    return blended.astype(np.uint8)

# Main process function
def process_images(model_path, content_image_path, design_patch_path, confidence_threshold=0.3, alpha=0.7):
    # Load the model
    model = load_model(model_path)

    # Load images
    content_image = Image.open(content_image_path).convert("RGB")
    design_patch = Image.open(design_patch_path).convert("RGB")

    # Process content image to get mask
    mask, image_np = get_dress_mask(model, content_image, confidence_threshold)
    if mask is None:
        return None

    fabric = cv2.cvtColor(np.array(design_patch), cv2.COLOR_RGB2BGR)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Overlay fabric on dress
    overlaid_image = overlay_fabric_on_dress(image_np, mask, fabric)

    # Blend images
    final_result = blend_images(image_np, overlaid_image, mask, alpha)
    return final_result

# Example usage
final_image = process_images("mask_rcnn_IS_50.pth",  "X:\\Sample convention images\\Dresses\\straight_trouser.png", "X:\\Sample convention images\\Dresses\\t4.jpg")
if final_image is not None:
    cv2.imwrite("stylized_dress.jpg", final_image)
