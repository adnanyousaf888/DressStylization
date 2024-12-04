import torch
import torchvision
import torchvision.transforms as T
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure  # To find contours
import cv2  # OpenCV for image processing
from skimage import transform  # For warping


# Reverse mapping for easy lookup
label_to_class = {v: k for k, v in class_mapping.items()}

# Function to load the model
def load_model(model_path, num_classes):
    from torchvision.models.detection import maskrcnn_resnet50_fpn

    # Since we're loading our own weights, set weights=None
    weights = None
    model = maskrcnn_resnet50_fpn(weights=weights)

    # Modify the model for your number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes)

    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

# Function to prepare the image for the model
def prepare_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform_fn = T.Compose([T.ToTensor()])  # Same transform used during training
    img_tensor = transform_fn(img)
    return img_tensor, img

# Function to get the dress mask
def get_dress_mask(model, image_path, confidence_threshold=0.3):
    # Prepare the image
    img_tensor, img_pil = prepare_image(image_path)

    # Add a batch dimension (batch size of 1)
    img_tensor = img_tensor.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        predictions = model(img_tensor)

    # Post-process the predictions
    pred_scores = predictions[0]['scores'].cpu().numpy()
    pred_labels = predictions[0]['labels'].cpu().numpy()
    pred_masks = predictions[0]['masks'].cpu().numpy()  # Shape: [N, 1, H, W]

    # Filter out predictions below the confidence threshold
    keep = pred_scores >= confidence_threshold

    pred_labels = pred_labels[keep]
    pred_scores = pred_scores[keep]
    pred_masks = pred_masks[keep]

    # Debugging: Print predicted labels and scores
    print("Predicted labels and scores:")
    for label, score in zip(pred_labels, pred_scores):
        class_name = class_mapping.get(label, 'Unknown')
        print(f"Label: {label} ({class_name}), Score: {score}")

    # Use the correct label for 'dress' (e.g., 'straight_shirt')
    dress_label = label_to_class['straight_shirt']  # Set to 7 based on your mapping

    # Find masks corresponding to the dress
    dress_masks = pred_masks[pred_labels == dress_label]

    if len(dress_masks) == 0:
        print("No dress detected in the image.")
        return None, None


    return combined_mask, image

# Function to overlay fabric pattern onto dress
def overlay_fabric_on_dress(image, mask, fabric):
    # Ensure the mask is binary
    mask = mask.astype(np.uint8)
    mask[mask > 0] = 1

    # Resize fabric to match the image size
    fabric_resized = cv2.resize(fabric, (image.shape[1], image.shape[0]))

    # Warp the fabric pattern to fit the dress shape
    # Get coordinates of the mask where mask == 1
    y_coords, x_coords = np.where(mask == 1)
    dress_points = np.column_stack((x_coords, y_coords))

    # Sample control points from the dress area
    num_control_points = 500  # Adjust for accuracy vs. performance
    

    # Apply the warped fabric to the dress area
    result = image.copy()
    for c in range(3):  # For each color channel
        result[:, :, c] = result[:, :, c] * (1 - mask) + warped_fabric[:, :, c] * mask

    return result

# Function to blend images for better realism
def blend_images(original_image, overlaid_image, mask, alpha=0.7):
    # Ensure mask is in the range [0, 1]
    mask = mask.astype(np.float32)
    mask = cv2.GaussianBlur(mask, (7, 7), 0) / 255.0  # Feather the mask edges
    return blended


# Main function
def main():
    # Paths to your model, image, and fabric pattern
    model_path = "C:\\Users\\outco\\Downloads\\Tomorrow Files\\mask_rcnn_IS_50.pth"  # Update this path
    image_path = "C:\\Users\\outco\\Downloads\\content_7.jpeg"  # Update this path
    fabric_path = "C:\\Users\\outco\\Downloads\\t4.jpg"  # Update this path

    # Update 'num_classes' to match your dataset
    num_classes = len(class_mapping) + 1  # +1 if background is considered separately

    # Load the model
    model = load_model(model_path, num_classes)

    # Get the dress mask and original image
    mask, image = get_dress_mask(model, image_path, confidence_threshold=0.3)

    if mask is None:
        return

    # Display the result
    plt.figure(figsize=(10, 10))
    plt.imshow(final_result)
    plt.axis('off')
    plt.show()

    # Optionally, save the result
    result_path = "C:\\Users\\outco\\Desktop\\stylized_dress.jpg"  # Update this path
    final_result_bgr = cv2.cvtColor(final_result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(result_path, final_result_bgr)

if __name__ == "__main__":
    main()
