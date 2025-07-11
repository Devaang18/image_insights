import os
import io
import json
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from google.cloud import vision
from google.oauth2 import service_account
from PIL import Image
import requests
from dotenv import load_dotenv
load_dotenv()

SERPER_KEY = os.environ["SERPER_KEY"]
gcp_creds_json = os.environ["GCP_CREDS"]
gcp_creds_dict = json.loads(gcp_creds_json)

credentials = service_account.Credentials.from_service_account_info(gcp_creds_dict)
client = vision.ImageAnnotatorClient(credentials=credentials)


app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def root():
    return RedirectResponse(url="/upload")

@app.get("/upload", response_class=HTMLResponse)
async def upload_form(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def handle_upload(request: Request, file: UploadFile = File(...)):
    content = await file.read()
    result = analyze_image(content)
    result_json = json.dumps(result, indent=2)
    return templates.TemplateResponse("upload.html", {"request": request, "result": result_json, "parsed": result})

@app.post("/analyze")
async def analyze_api(image: UploadFile = File(...)):
    content = await image.read()
    result = analyze_image(content)
    return JSONResponse(content=result)

def analyze_image(image_content):
    image = vision.Image(content=image_content)
    result = {}

    objects = client.object_localization(image=image).localized_object_annotations
    result["objects"] = [obj.name for obj in objects]

    text_response = client.text_detection(image=image)
    result["text"] = text_response.text_annotations[0].description if text_response.text_annotations else ""

    safe = client.safe_search_detection(image=image).safe_search_annotation
    result["nsfw"] = {
        "adult": safe.adult.name,
        "violence": safe.violence.name,
        "racy": safe.racy.name
    }

    logos = client.logo_detection(image=image).logo_annotations
    result["logos"] = [logo.description for logo in logos]

    props = client.image_properties(image=image).image_properties_annotation
    result["brand_colors"] = [
        {
            "rgb": {
                "r": int(color.color.red),
                "g": int(color.color.green),
                "b": int(color.color.blue)
            }
        }
        for color in props.dominant_colors.colors[:3]
    ]

    if result["objects"]:
        result["caption"] = "Image contains: " + ", ".join(result["objects"])
    else:
        result["caption"] = "No clear objects found"

    # Analyze face / identity
    person_info = analyze_face_identity(image_content)
    result["person"] = person_info

    # If a person is detected, fetch controversies
    if person_info and not person_info.get("error") and person_info.get("entity"):
        controversies = search_controversies_serper(person_info["entity"])
        result["controversies"] = controversies
    else:
        result["controversies"] = []

    return result


def crop_face(image_content):
    image = vision.Image(content=image_content)
    face_response = client.face_detection(image=image)
    faces = face_response.face_annotations

    if not faces:
        return None 

    box = faces[0].bounding_poly.vertices
    im = Image.open(io.BytesIO(image_content))

    left = box[0].x
    top = box[0].y
    right = box[2].x
    bottom = box[2].y

    cropped_face = im.crop((left, top, right, bottom))

    buf = io.BytesIO()
    cropped_face.save(buf, format='JPEG')
    return buf.getvalue()

def detect_web_entities(image_content):
    image = vision.Image(content=image_content)
    web_response = client.web_detection(image=image)
    web = web_response.web_detection

    results = {}

    if web.best_guess_labels:
        results["best_guess"] = web.best_guess_labels[0].label

    if web.web_entities:
        results["entity"] = web.web_entities[0].description

    return results

def analyze_face_identity(image_content):
    cropped = crop_face(image_content)
    if not cropped:
        return {"error": "No face detected"}
    return detect_web_entities(cropped)

def search_controversies_serper(person_name):
    url = "https://google.serper.dev/search"
    query = f"Latest {person_name} controversies 2025 - descriptive headlines"

    headers = {
        "X-API-KEY": SERPER_KEY,
        "Content-Type": "application/json"
    }

    data = {"q": query}
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        result = response.json()
        organic = result.get("organic", [])
        controversies = []
        for item in organic:
            title = item.get("title")
            snippet = item.get("snippet")
            link = item.get("link")
            controversies.append({
                "title": title,
                "snippet": snippet,
                "link": link
            })
        return controversies
    else:
        return []