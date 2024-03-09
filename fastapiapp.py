from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from typing import Annotated
from PIL import Image
import io
import torch
from torchvision import transforms
from torch.nn import functional as F
import os
from torchvision.models import VGG16_Weights

app = FastAPI()
# UPLOAD_DIR = "app/app/static/uploads"
# app.mount("/app/app/static", StaticFiles(directory="static"), name="static")
def load_vgg16_model():
    return torch.jit.load('vgg16_final_model_jit.pt').to('cpu')

def image_transform():
    weights = VGG16_Weights.DEFAULT
    auto_transforms = weights.transforms()
    return auto_transforms

def classify_image(model, image):
    transform = image_transform()
    img_transformed = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output_logits = model(img_transformed)

    probabilities = F.softmax(output_logits, dim=1).squeeze(dim=0)
    return torch.argmax(probabilities).item(), probabilities.tolist()

def inference(file):
    vgg16_model = torch.jit.load('vgg16_final_model_jit.pt').to('cpu')
    vgg16_model.eval()

    img = Image.open(io.BytesIO(file)).convert('RGB')
    transform = image_transform()
    img = transform(img)
    img = img.unsqueeze(0)

    with torch.no_grad():
        output_logits = vgg16_model(img)

    probabilities = F.softmax(output_logits, dim=1)
    predicted_class = torch.argmax(probabilities).item()
    probabilities = probabilities.squeeze(dim=0).tolist()
    
    class_labels = ["Normal", "Pneumonia"]
    return {"Predicted Class": predicted_class, "Class Label": class_labels[predicted_class], "probabilities": probabilities}

def delete_uploaded_image(image_path: str):
    os.remove(image_path)
    print(f"Image deleted: {image_path}")

@app.get("/")
def first():
    return {'message':"Hello World"}

@app.post("/infer/")
async def create_upload_file(file: UploadFile = File(...)):
    result = inference(await file.read())
    return JSONResponse(content=result)
