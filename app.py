import os
from datetime import datetime, timedelta
import fitz
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas
from flask import Flask, request
from langchain_core.messages import HumanMessage
from langchain_core.utils import get_from_env
from langchain_openai import AzureChatOpenAI
from configuration import get_configuration
from file_storage import Document

app = Flask(__name__)

def init_document_analysis_client():
    configuration = get_configuration("azure")

    return DocumentIntelligenceClient(
        endpoint=configuration["cognitive_services_url"],
        credential=_get_azure_credentials(),
    )


def _get_azure_credentials():
    key = get_from_env("azure_cogs_key", "DI_KEY")
    return AzureKeyCredential(key)


def init_azure_chat() -> AzureChatOpenAI:
    configuration = get_configuration("azure")

    return AzureChatOpenAI(
        azure_endpoint=configuration["openai_api_base"],
        openai_api_version=configuration["openai_api_version"],
        deployment_name=configuration["deployment_name"],
        temperature=0
    )

def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    converted_images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        image_path = f"page_{page_num}.png"
        pix.save(image_path)
        converted_images.append(image_path)
    return converted_images

def upload_to_azure(converted_images):
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    container_client = blob_service_client.get_container_client('image-upload')
    image_urls = []
    for image in converted_images:
        blob_client = container_client.get_blob_client(os.path.basename(image))
        with open(image, "rb") as data:
            blob_client.upload_blob(data=data, overwrite=True)

        sas_token = generate_blob_sas(
            account_name = blob_service_client.account_name,
            container_name = 'image-upload',
            blob_name = os.path.basename(image),
            account_key = blob_service_client.credential.account_key,
            permission = BlobSasPermissions(read=True),
            expiry = datetime.now() + timedelta(minutes=5)  # Set expiry time as needed
        )
        blob_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/image-upload/{os.path.basename(image)}?{sas_token}"
        image_urls.append(blob_url)
    return image_urls


def analyze_images(image_urls):
    # Combine all image URLs into a single prompt
    prompt = "Analyze the following images and extract basic and pricing information. Return the result as key value pairs in json format."
    image_urls_content = [{"type": "image_url", "image_url": {"url": url}} for url in image_urls]

    chatClient = init_azure_chat()

    message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                *image_urls_content
            ]
        )

    response = chatClient.invoke([message])
    return response.content

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

@app.route('/analyse', methods=['POST'])
def analyse():
    with Document(request.files['file']) as pdf:
        images = pdf_to_images(pdf.filename)
        image_urls = upload_to_azure(images)
        analysis_results = analyze_images(image_urls)

    return analysis_results

if __name__ == '__main__':
    app.run()
