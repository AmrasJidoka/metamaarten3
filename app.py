import base64
import io

import fitz
from PIL import Image
from flask import Flask, request, jsonify
from langchain_openai import AzureChatOpenAI
from pdf2image import convert_from_bytes

from configuration import get_configuration

app = Flask(__name__)

def init_azure_chat() -> AzureChatOpenAI:
    configuration = get_configuration("azure")

    return AzureChatOpenAI(
        azure_endpoint=configuration["openai_api_base"],
        openai_api_version=configuration["openai_api_version"],
        deployment_name=configuration["deployment_name"],
        temperature=0
    )

def pdf_to_images(pdf_bytes):
    images = convert_from_bytes(pdf_bytes)
    return images

def send_images_to_llm(images):
    client = init_azure_chat()

    images_byte_array = []
    for image in images:
        images_byte_array = io.BytesIO()
        image.save(images_byte_array, format='PNG')
        images_byte_array.seek(0)
        images_byte_array.write(images_byte_array.read())

    response = client.chat(images=images_byte_array)
    return response

def pdf_page_to_base64(file_stream):
    pdf_document = fitz.open(stream=file_stream, filetype='pdf')
    images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes()))
        img_byte_array = io.BytesIO()
        img.save(img_byte_array, format='PNG')
        img_byte_array.seek(0)
        images.append(img_byte_array.read())

    return jsonify({"message": "Images converted and kept in memory"}), 200

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route('/analyse', methods=['POST'])
def analyse():
    file = request.files['file']
    pdf_bytes = file.read()
    processed_images = pdf_to_images(pdf_bytes)
    response = send_images_to_llm(processed_images)
    return jsonify(response)

if __name__ == '__main__':
    app.run()
