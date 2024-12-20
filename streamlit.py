from ultralytics import YOLO
import cv2
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import streamlit as st
from PIL import Image
import torch
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import gdown
import os


@st.cache_resource
def download_model():
    file_id = "18M5NP0p4a56iwXqCwunT4m-CemftzGDp"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "model.pt"

    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

    return output

faster_rcnn_model_path = download_model()

def yolo_inference(image, model_path="best.pt"):
    model = YOLO(model_path)
    
    results = model(image)
    
    annotated_image = results[0].plot()
    detections = results[0].boxes
    
    return annotated_image, detections

def faster_rcnn_inference(image, model_path=None, device="cpu"):
    model = fasterrcnn_resnet50_fpn(pretrained=(model_path is None))
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    input_image = F.to_tensor(image).unsqueeze(0).to(device)
    
    outputs = model(input_image)[0]
    
    boxes = outputs["boxes"].cpu().detach().numpy()
    scores = outputs["scores"].cpu().detach().numpy()
    labels = outputs["labels"].cpu().detach().numpy()
    
    for box, score in zip(boxes, scores):
        if score > 0.3:
            x_min, y_min, x_max, y_max = map(int, box)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, f"{score:.2f}", (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image, {"boxes": boxes, "scores": scores, "labels": labels}

def main():
    st.title("Object Detection App")
    st.sidebar.title("Model Selection")
    st.sidebar.write("Choose a model for object detection.")
    
    model_option = st.sidebar.selectbox("Select Model", ("YOLOv8", "Faster R-CNN"))

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        image_np = np.array(image)
        
        if st.button("Run Detection"):
            if model_option == "YOLOv8":
                st.write("Running YOLOv8...")
                model_path = "best_yolo_model.pt"
                annotated_image, detections = yolo_inference(image_np, model_path)
                st.image(annotated_image, caption="YOLOv8 Detection", use_container_width=True)
                st.write(detections)
            
            elif model_option == "Faster R-CNN":
                st.write("Running Faster R-CNN...")
                model_path = faster_rcnn_model_path
                annotated_image, detections = faster_rcnn_inference(image_np, model_path=faster_rcnn_model_path, device="cpu")
                st.image(annotated_image, caption="Faster R-CNN Detection", use_column_width=True)
                st.write(detections)

if __name__ == "__main__":
    main()

#streamlit run app.py
