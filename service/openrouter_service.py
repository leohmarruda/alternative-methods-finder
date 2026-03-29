import os
from pathlib import Path

import requests

DEFAULT_MODEL = "openai/gpt-4o"

_MOCK_OUTPUT_PATH = Path(__file__).resolve().parent / "mock_output.md"


def mock_response(_prompt: str) -> str:
    return _MOCK_OUTPUT_PATH.read_text(encoding="utf-8")


def open_router(prompt, model=DEFAULT_MODEL):

    contexto = """
    You are a specialized Bio-Informatics Assistant. Your task is to parse "Materials and Methods" sections from biomedical research papers and extract animal-based experimental protocols into a structured JSON format. You are an expert in the 3Rs (Replacement, Reduction, Refinement) and OECD Test Guidelines.
    Please analyze the provided "Materials and Methods" text. Your goal is to identify and extract every distinct animal experiment, particularly those related to General Toxicology, Carcinogenicity, Skin Irritation/Corrosion, and Eye Irritation.

    For each experiment found, output the following JSON schema:
    
    section_id: The heading number from the text (e.g., "3.5.1").
    
    domain: Categorize as: 'Acute Tox', 'Repeated Dose', 'Carcinogenicity', 'Skin Irritation', 'Eye Irritation', or 'Other'.
    
    test_description: A brief summary of the procedure (e.g., "Draize eye test" or "28-day oral gavage").
    
    species: The animal model used (e.g., Rabbit, Wistar Rat, BALB/c Mouse).
    
    sample_size: Total number of animals used in this specific test.
    
    administration_route: (e.g., p.o., i.p., topical, inhalation).
    
    endpoints_measured: List specific biological markers (e.g., Erythema, LD50, ALT/AST levels, Tumor incidence).
    
    duration: Total timeframe (e.g., 72 hours, 2 years).
    
    is_regulatory_standard: Boolean (True if it mentions OECD, ISO, or Pharmacopoeia standards).
    
    Processing Rules:
    
    If a section describes multiple tests (e.g., both Skin and Eye irritation), create separate JSON objects for each.
    
    Look for "Trigger Keywords":
    
    Skin: Erythema, edema, shaved, patch, corrosion, Draize.
    
    Eye: Conjunctival, corneal opacity, iris, instillation, lacrimation.
    
    Carcinogenicity: Neoplasm, adenoma, tumor, chronic 2-year, malignancy.
    
    Ignore non-animal sections like Chemical Analysis, Plant Extraction, or Statistical Methods.
    
    Input Text:
    {user_input}
    """
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ.get('API_KEY')}"
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Você é um especialista e deve responder apenas com base no contexto fornecido."
                },
                {
                    "role": "user",
                    "content": f"""
                    Contexto:
                    {contexto}
    
                    Pergunta:
                    {prompt}
                    """
                }
            ]
        }
    )

    data = response.json()
    return data['choices'][0]['message']['content']
