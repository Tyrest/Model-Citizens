import sys

sys.path.append(".")
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import box_convert
from transformers import LxmertForQuestionAnswering, LxmertTokenizer

from load_nlvr import load_nlvr

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)


train_df, dev_df, test_df = load_nlvr()

# Load pre-trained Faster R-CNN model
faster_rcnn = fasterrcnn_resnet50_fpn(pretrained=True)
faster_rcnn.eval()


# Function to extract visual features
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        prediction = faster_rcnn(image)

    boxes = prediction[0]["boxes"]
    features = prediction[0]["features"]

    # Normalize boxes
    boxes = box_convert(boxes, in_fmt="xyxy", out_fmt="xyxy")
    boxes = boxes / torch.tensor([224, 224, 224, 224])

    return features, boxes


# Load LXMERT model and tokenizer
tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
model = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-base-uncased")


# Prepare inputs
sentence = dev_df[0]["sentence"]
left_image_path = dev_df[0]["left"]
right_image_path = dev_df[0]["right"]

# Extract features for both images
left_features, left_boxes = extract_features(left_image_path)
right_features, right_boxes = extract_features(right_image_path)

# Combine features and boxes
visual_feats = torch.cat([left_features, right_features], dim=0)
visual_pos = torch.cat([left_boxes, right_boxes], dim=0)

# Tokenize the sentence
inputs = tokenizer(sentence, return_tensors="pt")

# Prepare inputs for LXMERT
inputs.update(
    {
        "visual_feats": visual_feats,
        "visual_pos": visual_pos,
    }
)

# Pass inputs to the model
outputs = model(**inputs)
