import sys

sys.path.append(".")

import detectron2
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.structures.boxes import Boxes
from transformers import AutoTokenizer, VisualBertForVisualReasoning

from load_nlvr import load_nlvr


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


train_df, dev_df, test_df = load_nlvr()

# Configure a Faster R-CNN model similar to what was likely used in the paper
cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # Lower threshold to get more proposals
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = "detectron2://ImageNetPretrained/MSRA/R-101.pkl"
predictor = DefaultPredictor(cfg)


def extract_features(image, max_proposals=144):
    """
    Extract region-based visual features from an image.

    Args:
        image: The input image (numpy array, BGR format)
        max_proposals: Maximum number of region proposals to use

    Returns:
        visual_features: Tensor of shape (num_proposals, feature_dim)
        boxes: Bounding box coordinates (num_proposals, 4)
    """
    # Run inference with Detectron2
    with torch.no_grad():
        outputs = predictor(image)

    # Get features, boxes, and scores
    features = outputs[
        "roi_features"
    ]  # You may need to modify predictor to output these
    instances = outputs["instances"]
    boxes = instances.pred_boxes.tensor
    scores = instances.scores

    # Sort by confidence and take top proposals
    sorted_idxs = torch.argsort(scores, descending=True)
    if len(sorted_idxs) > max_proposals:
        sorted_idxs = sorted_idxs[:max_proposals]

    boxes = boxes[sorted_idxs]
    features = features[sorted_idxs]

    return features, boxes


sentence = dev_df[0]["sentence"]
left_image_path = dev_df[0]["left"]
right_image_path = dev_df[0]["right"]

# Extract features for both images
left_features, left_boxes = extract_features(left_image_path)
right_features, right_boxes = extract_features(right_image_path)
