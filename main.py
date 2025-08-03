from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
import os
import re
import uuid
from PIL import Image

app = FastAPI()

# Clean up text
def clean(text):
    return text.strip().replace("�", "")

# Extract key fields using regex
def extract_fields(text):
    fields = {
        'National_ID': re.search(r'National ID\s+(\d{10,})', text),
        'Pin': re.search(r'Pin\s+(\d{14,})', text),
        'Name_Bangla': re.search(r'Name\(Bangla\)\s+(.+)', text),
        'Name_English': re.search(r'Name\(English\)\s+(.+)', text),
        'Date_of_Birth': re.search(r'Date of Birth\s+([\d-]+)', text),
        'Birth_Place': re.search(r'Birth Place\s+(.+)', text),
        'Father_Name': re.search(r'Father Name\s+(.+)', text),
        'Mother_Name': re.search(r'Mother Name\s+(.+)', text),
        'Spouse_Name': re.search(r'Spouse Name\s+(.+)', text),
        'Religion': re.search(r'Religion\s+(.+)', text),
        'Dob': re.search(r'Date of Birth\s+(.+)', text),
        'blood': re.search(r'Blood Group\s+(.+)', text),
        'Gender': re.search(r'Gender\s+(.+)', text),
        'Marital_Status': re.search(r'Marital\s+(.+)', text),
        'Occupation': re.search(r'Occupation\s+(.+)', text),
        'Education': re.search(r'Education\s+(.+)', text),
    }

    address = {
        'Division': re.search(r'Permanent Address Division\s+(.+)', text),
        'District': re.search(r'District\s+(.+)', text),
        'Upozila': re.search(r'Upozila\s+(.+)', text),
        'Union_Ward': re.search(r'Union/Ward\s+(.+)', text),
        'Mouza': re.search(r'Mouza/Moholla\s+(.+)', text),
        'Village_Road': re.search(r'Village/Road\s+(.+)', text),
        'Post_Office': re.search(r'Post Office\s+(.+)', text),
        'Postal_Code': re.search(r'Postal Code\s+(\d+)', text),
        'Region': re.search(r'Region\s+(.+)', text),
    }

    data = {}
    for k, v in fields.items():
        data[k] = clean(v.group(1)) if v else None

    data['Permanent_Address'] = {}
    for k, v in address.items():
        data['Permanent_Address'][k] = clean(v.group(1)) if v else None

    return data

@app.post("/extract/")
async def extract_pdf(pdf_file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    pdf_path = f"uploads/{uid}.pdf"

    with open(pdf_path, "wb") as f:
        f.write(await pdf_file.read())

    doc = fitz.open(pdf_path)
    text = ""
    image_paths = []

    for page_num, page in enumerate(doc):
        text += page.get_text()
        for i, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            img_path = f"images/{uid}_page{page_num + 1}_img{i + 1}.png"
            if pix.n < 5:
                pix.save(img_path)
            else:
                pix1 = fitz.Pixmap(fitz.csRGB, pix)
                pix1.save(img_path)
                pix1 = None
            pix = None
            image_paths.append(img_path)

    doc.close()

    # Heuristic face and signature detection
    classified = {"face_image": None, "signature_image": None}
    for img in image_paths:
        try:
            with Image.open(img) as im:
                w, h = im.size
                ratio = w / h
                if 0.7 < ratio < 1.5 and h > 100:
                    if not classified["face_image"]:
                        classified["face_image"] = img
                elif ratio > 2.5 and h < 100:
                    if not classified["signature_image"]:
                        classified["signature_image"] = img
        except:
            continue

    result = extract_fields(text)
    result.update(classified)

    return JSONResponse(result)
