import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

# Configuration - Update these paths as needed
MODEL_DIR = './models'  # Default local directory
MODEL_PATH = os.path.join(MODEL_DIR, 'best_gcn_model0712V1.pth')
IMAGE_FOLDER = os.path.join(MODEL_DIR, 'IMG')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VEHICLE_CLASSES = ["car", "non-car"]

class InferenceGCN(torch.nn.Module):
    """Dynamically built classifier head matching the GCN checkpoint."""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.fc(x)

def load_model(model_path):
    """Load just the classifier head from the GCN checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please update the MODEL_PATH variable.")

    print(f"Loading model from {model_path}...")
    ckpt = torch.load(model_path, map_location=DEVICE)
    state_dict = ckpt.get('model_state_dict', ckpt)

    # Extract the weight tensors for dimensions
    w0 = state_dict['cls_head.0.weight']
    w1 = state_dict['cls_head.3.weight']

    in_dim = w0.size(1)
    hidden_dim = w0.size(0)
    out_dim = w1.size(0)

    model = InferenceGCN(in_dim, hidden_dim, out_dim).to(DEVICE)

    # Copy weights & biases
    model.fc[0].weight.data.copy_(state_dict['cls_head.0.weight'])
    model.fc[0].bias.data.copy_(state_dict['cls_head.0.bias'])
    model.fc[2].weight.data.copy_(state_dict['cls_head.3.weight'])
    model.fc[2].bias.data.copy_(state_dict['cls_head.3.bias'])

    model.eval()
    return model

def setup_detectron2():
    """Setup Detectron2 config & predictor."""
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)

def process_image(image_path, classifier_model, predictor):
    """Run Detectron2 Mask R-CNN then classify each detected vehicle via GCN head."""
    # Read image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None, []
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Detect instances
    outputs = predictor(img_bgr)
    instances = outputs["instances"].to("cpu")

    # Filter vehicle classes (COCO IDs: car=2, motorcycle=3, airplane=4, bus=6, truck=8)
    vehicle_idxs = [i for i, c in enumerate(instances.pred_classes) if int(c) in [2, 3, 4, 6, 8]]
    
    if not vehicle_idxs:
        return img_rgb, []

    car_masks = []
    feat_dim = classifier_model.fc[0].in_features

    for idx in vehicle_idxs:
        mask = instances.pred_masks[idx].numpy().astype(np.uint8)
        bbox = instances.pred_boxes[idx].tensor.numpy()[0]

        # Dummy features matching the head’s expected input size
        # NOTE: Ideally replaces with actual extracted features
        features = torch.randn(1, feat_dim, device=DEVICE)

        with torch.no_grad():
            logits = classifier_model(features)
            probs = torch.softmax(logits, dim=1)[0]

        # If “car” (index 0) is top prediction
        if torch.argmax(probs).item() == 0:
            car_masks.append({
                'mask': mask,
                'bbox': bbox,
                'score': probs[0].item()
            })

    return img_rgb, car_masks

def visualize_results(img, car_masks):
    """Overlay red masks & blue bboxes with scores on the RGB image."""
    disp = img.copy()
    for item in car_masks:
        mask = item['mask']
        x1, y1, x2, y2 = item['bbox'].astype(int)

        # Draw red mask overlay
        color_mask = np.zeros_like(disp)
        color_mask[..., 0] = mask * 255  # Red channel
        disp = np.where(mask[..., None],
                        cv2.addWeighted(disp, 0.7, color_mask, 0.3, 0),
                        disp)
        
        # Draw bounding box (Blue)
        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw label text
        label = f"Car: {item['score']:.2f}"
        cv2.putText(disp, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return disp

if __name__ == "__main__":
    # Ensure image folder exists
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Image folder not found: {IMAGE_FOLDER}")
        exit()

    try:
        classifier = load_model(MODEL_PATH)
        predictor = setup_detectron2()
        
        print("Processing images...")
        image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        if not image_files:
            print("No images found in folder.")
        
        for fname in image_files:
            img_path = os.path.join(IMAGE_FOLDER, fname)
            img, masks = process_image(img_path, classifier, predictor)
            
            if img is not None:
                result = visualize_results(img, masks)
                
                plt.figure(figsize=(12, 8))
                plt.imshow(result)
                plt.axis('off')
                plt.title(f"Car Segmentation: {fname}")
                plt.show()
                print(f"Displayed results for {fname}")

    except Exception as e:
        print(f"Error: {e}")
