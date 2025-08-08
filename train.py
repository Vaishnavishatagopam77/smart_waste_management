# Importing the YOLO class from the ultralytics library to use YOLOv8
from ultralytics import YOLO

# Optional import: used if you want to validate/export the model in ONNX format later
import onnx

# This line initializes a new YOLO model architecture using the 'yolov8n.yaml' config file.
# 'yolov8n' stands for YOLOv8-Nano – a very lightweight version of YOLOv8.
model = YOLO('yolov8n.yaml')

# This line instead loads a pre-trained YOLOv8n model from a .pt file.
# You’ll typically use this if you want to fine-tune or perform inference using an already-trained model.
model = YOLO('yolov8n.pt')

# Set the path to your dataset configuration file (data.yaml) which contains:
# - class names
# - paths to training and validation image folders
# - number of classes
path = '/content/datasets/waste-detection-9/data.yaml'

# Start training the model using the dataset defined in data.yaml
# Runs for 50 epochs (i.e., 50 complete passes over the dataset)
results = model.train(data=path, epochs=50)

# After training, this line runs evaluation on the validation set to measure performance (e.g., mAP, precision, recall)
results = model.val()

# Exports the trained model to ONNX format so that it can be used in other environments like:
# web apps, mobile devices, or edge devices like Raspberry Pi or Jetson Nano
success = model.export(format='onnx')
