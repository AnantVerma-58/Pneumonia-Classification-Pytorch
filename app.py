from fastapi import FastAPI, Path, HTTPException, File, Query,UploadFile, Request, Depends, BackgroundTasks, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
import shutil
import torch
from torchvision import transforms
from torch.nn import functional as F
import os
from model_inference import inference

UPLOAD_DIR = "static"
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/static/uploads", StaticFiles(directory="static/uploads"), name="uploads")
from torchvision.models import VGG16_Weights

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

# Load the VGG16 model
# vgg16_model = 

templates = Jinja2Templates(directory="templates")
class_labels = ["Normal", "Pneumonia"]

def delete_uploaded_image(image_path: str):
    import time
    time.sleep(100)  # Delay for 100 seconds
    os.remove(image_path)
    print(f"Image deleted: {image_path}")


async def get_template_context(request: Request):
    return {"request": request}


# @app.get("/")
# def read_form(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/predict")
# async def predict(request: Request,
#                   background_tasks: BackgroundTasks,
#                   image_source: str = Form(None),
#                   sample_image: str = Form(None),
#                   file: UploadFile = File(None),
#                   template_context: dict = Depends(get_template_context)):
#      print("Hi")
#      print(image_source)
#      try:
#         if image_source == "sample":
#             selected_image = os.path.join("static", f"{sample_image}.jpeg")
#         else:
#             # Use the uploaded file
#             contents = await file.read()
#             image = Image.open(io.BytesIO(contents)).convert('RGB')
#             selected_image = os.path.join(UPLOAD_DIR, file.filename)

#             # Save the uploaded image
#             with open(selected_image, "wb") as f:
#                 f.write(contents)

#         predicted_class, probabilities = classify_image(vgg16_model, image)

#         template_context.update({
#             "uploaded_filename": selected_image,
#             "prediction": class_labels[predicted_class],
#             "probabilities": str(probabilities),
#         })

#         background_tasks.add_task(os.remove, selected_image)
#         print(template_context)
#         return templates.TemplateResponse("result.html", template_context, status_code=200)

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/", response_class=HTMLResponse)
async def home(request: dict = {}):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/uploadfile", response_class=HTMLResponse)
async def create_upload_file(background_tasks: BackgroundTasks,
                             file: UploadFile = File(...),
                             template_context: dict = Depends(get_template_context)):
    # Save the uploaded file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    selected_image = os.path.join(f"{UPLOAD_DIR}/uploads", file.filename)
    with open(selected_image, "wb") as f:
        f.write(contents)
    
    infer = inference(image_path=selected_image)
    proabiblites, class_label = infer["probabilities"], infer["Class Label"]
    # os.remove(selected_image)
    # print(proabiblites, class_label)
    # print("HI")
    template_context.update({"uploaded_filename":f"/static/uploads/{file.filename}",
                             "prediction":class_label,
                             "probabilities":proabiblites})
    # print(template_context)
    # os.remove(selected_image)
    background_tasks.add_task(delete_uploaded_image, selected_image, delay=100)
    return templates.TemplateResponse("result.html", context=template_context, status_code=200)


# @app.get("/sample", response_class=HTMLResponse)
# async def show_sample(sample_image: str = Form(...),
#                       sample_number: int = Path(..., title="Sample Number"),
#                       template_context:dict=Depends(get_template_context)):
#     infer = inference(image_path=f"static/samples/sample{sample_number}.jpeg")
#     proabiblites, class_label = infer["probabilities"], infer["Class Label"]
#     # os.remove(selected_image)
#     print(proabiblites, class_label)
#     # print("HI")
#     template_context.update({"uploaded_filename":f"/static/samples/sample{sample_number}.jpeg",
#                              "prediction":class_label,
#                              "probabilities":proabiblites})
#     # print(template_context)
#     # os.remove(selected_image)
#     return templates.TemplateResponse("sample.html", context=template_context, status_code=200)


@app.post("/sample", response_class=HTMLResponse)
async def show_sample(sample_image: str = Form(...), template_context: dict = Depends(get_template_context)):
    # Extract the sample number from the selected image string
    sample_number = int(sample_image.split()[-1])

    infer = inference(image_path=f"static/samples/sample{sample_number}.jpeg")
    probabilities, class_label = infer["probabilities"], infer["Class Label"]

    template_context.update({
        "uploaded_filename": f"/static/samples/sample{sample_number}.jpeg",
        "prediction": class_label,
        "probabilities": probabilities
    })

    return templates.TemplateResponse("result.html", context=template_context, status_code=200)