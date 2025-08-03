from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
import os
import re
import uuid
from PIL import Image
import base64
from io import BytesIO

app = FastAPI()


def clean(text):
    return text.strip().replace("�", "")


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
        'blood': re.search(r'Blood Group\s+(.+)', text),
        'Gender': re.search(r'Gender\s+(.+)', text),
        'Marital_Status': re.search(r'Marital\s+(.+)', text),
        'Occupation': re.search(r'Occupation\s+(.+)', text),
        'Education': re.search(r'Education\s+(.+)', text),
    }

    data = {}
    for k, v in fields.items():
        data[k] = clean(v.group(1)) if v else None

    # Extract Division with better logic
    lines = text.splitlines()
    division = None
    for i, line in enumerate(lines):
        if re.search(r'Permanent Address\s*Division', line, re.I):
            parts = line.split(':')
            if len(parts) > 1 and parts[1].strip():
                division = parts[1].strip()
            elif i + 1 < len(lines):
                division = lines[i + 1].strip()
            break

    address = {
        'Division': division,
        'District': None,
        'Upozila': None,
        'Union_Ward': None,
        'Mouza': None,
        'Village_Road': None,
        'Post_Office': None,
        'Postal_Code': None,
        'Region': None,
    }

    # Regex for the other address fields (ignore Division here)
    address_patterns = {
        'District': r'District\s+(.+)',
        'Upozila': r'Upozila\s+(.+)',
        'Union_Ward': r'Union/Ward\s+(.+)',
        'Mouza': r'Mouza/Moholla\s+(.+)',
        'Village_Road': r'Village/Road\s+(.+)',
        'Post_Office': r'Post Office\s+(.+)',
        'Postal_Code': r'Postal Code\s+(\d+)',
        'Region': r'Region\s+(.+)',
    }

    for key, pattern in address_patterns.items():
        match = re.search(pattern, text)
        address[key] = clean(match.group(1)) if match else None

    data['Permanent_Address'] = address

    # Construct full address by joining all available parts
    address_parts = [v for v in address.values() if v]
    data['full_address'] = ', '.join(address_parts)

    return data


def encode_image_base64(img_path):
    with Image.open(img_path) as im:
        buffered = BytesIO()
        im.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")


@app.post("/extract/")
async def extract_pdf(pdf_file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    pdf_path = f"uploads/{uid}.pdf"

    content = await pdf_file.read()
    with open(pdf_path, "wb") as f:
        f.write(content)

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
    os.remove(pdf_path)

    # Debug: print extracted text length and sample
    print(f"Extracted text length: {len(text)}")
    print(f"Extracted text sample:\n{text[:500]}")  # first 500 chars

    # Heuristic face and signature detection with logging
    classified = {"face_image": None, "signature_image": None}
    for img in image_paths:
        try:
            with Image.open(img) as im:
                w, h = im.size
                ratio = w / h
                print(f"Image {img}: width={w}, height={h}, ratio={ratio:.2f}")

                # Face heuristic: roughly square and tall enough
                if 0.7 < ratio < 1.4 and h > 100:
                    if not classified["face_image"]:
                        classified["face_image"] = encode_image_base64(img)

                # Signature heuristic: wide and relatively short
                elif ratio > 2.0 and 30 < h < 220:
                    if not classified["signature_image"]:
                        classified["signature_image"] = encode_image_base64(img)
        except Exception as e:
            print(f"Error processing image {img}: {e}")
            continue

    # Fallback: last image as signature if none found
    if not classified["signature_image"] and len(image_paths) > 0:
        classified["signature_image"] = encode_image_base64(image_paths[-1])

    result = extract_fields(text)
    result.update(classified)

    return JSONResponse(result)
