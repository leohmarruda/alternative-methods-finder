"""OpenRouter chat completion for protocol JSON extraction."""

from __future__ import annotations

import os
from pathlib import Path

import requests

DEFAULT_MODEL = "openai/gpt-4o"

_MOCK_OUTPUT_PATH = Path(__file__).resolve().parent / "mock_output.md"

SYSTEM_PROMPT = """You are a specialized Bio-Informatics Assistant. Your task is to interpret user input that may be either a full "Materials and Methods" excerpt **or** a short plain-language description of a study, disease area, or endpoint (e.g. a few words like "rabies diagnosis"). From that input, extract structured JSON for **in vivo** animal experiments, **ex vivo** tissue protocols, **in vitro** studies using animal-derived materials, and **in silico** methods where they appear or are plausibly relevant. You are an expert in the 3Rs (Replacement, Reduction, Refinement) and OECD Test Guidelines. Never refuse for brevity: always return valid JSON. Use null, "not stated", and empty arrays where details are missing; do not invent specific numbers, strains, or citations. You must respond with a single valid JSON object whose top-level keys are exactly "study_summary" and "experiments" as specified in the user instructions—no markdown outside the JSON."""

PROTOCOL_EXTRACTOR_PROMPT_TEMPLATE = '''Please analyze the user input below. It may be a full "Materials and Methods" passage **or** a short phrase or topic (e.g. "rabies diagnosis", "shampoo eye irritation"). Your goal is to (1) summarize what is known or implied for an ethics / IACUC-style reader, and (2) list every distinct **animal experiment** or **ex vivo / in vitro validation study** you can reasonably infer from the text—or, if the input is only a topic with no explicit protocol, produce **at most one** cautious placeholder experiment in domain **"Other"** with **test_description** reflecting the topic and almost all other fields **null** or **"not stated"** unless the input clearly specifies them.

## Short or underspecified input

- **Always** return the required JSON shape; do **not** ask for more text or apologize for lack of detail.
- In **study_summary**, state clearly when the input was brief or topic-only, what you inferred at a high level, and that specifics were not provided.
- Do **not** fabricate guideline numbers, n sizes, or proprietary methods; prefer sparse objects over invented detail.

## Detection scope

Include both **in vivo** (live animal) and **ex vivo / in vitro** experiments that use **animal-derived** biological material (e.g. slaughterhouse-obtained organs, isolated corneas, perfusion chambers, primary cultures, reconstructed human/epidermal models where the protocol is toxicity-related). **Model identification:** if the protocol uses isolated tissues from a slaughterhouse, a perfusion chamber, explants, or organ-on-a-chip with animal-derived tissue, treat it as its own experiment and set **model_type** to **"ex vivo"** (or **"in vitro"** for cell-line / culture-dish work without whole organs). Use **"in silico"** only for purely computational validation blocks that are clearly framed as a substitute or prediction method.

## Required output shape

Return **one JSON object** with exactly these top-level keys:

1. **"study_summary"** (string, 3–6 sentences, plain text):
   - Name the **test substance, product, or material** if stated (e.g. drug, essential oil, extract, **including drops or formulations in ex vivo eye models**).
   - State **species, strain, and approximate group sizes** (or **n** of explants / corneas / cultures) when available.
   - List the **main toxicological domains** and whether work is **in vivo vs ex vivo / in vitro**.
   - Note any **regulatory framing** (OECD, GLP, guideline numbers) if mentioned.
   - If the excerpt is silent on a point, say so briefly instead of inventing details.

2. **"experiments"** (array of objects): one object per distinct experiment. Each object must include **all** keys below (use null for unknown text fields, false for booleans when not applicable, [] for empty lists):

section_id: Heading number (e.g. "3.5.1") or a short label.

domain: Toxicology category — one of: 'Acute Tox', 'Repeated Dose', 'Carcinogenicity', 'Skin Irritation', 'Eye Irritation', 'Genotoxicity', 'Developmental / Reproductive', or 'Other'.

model_type: Exactly one of: **"in vivo"**, **"ex vivo"**, **"in vitro"**, **"in silico"**.

category: Same high-level label as **domain** (for reporting tables).

is_alternative_method: Boolean — **true** if this block already uses a **3Rs-style** approach: ex vivo, in vitro, slaughterhouse-sourced tissue, reconstructed tissue, organ-on-a-chip, or other **non–live-animal** test system for that endpoint; **false** for conventional **in vivo** animal tests.

test_description: Brief procedure summary (e.g. "Draize eye test", "BCOP", "EVEIT perfusion cornea", "28-day oral gavage").

species: For **in vivo**, species as usual (e.g. Rabbit, rat). For **ex vivo** / slaughterhouse / isolated organ studies, use the form **"Ex vivo [Species]"** (e.g. **"Ex vivo Bovine"**). For **in vitro** cell-only systems, species of origin if stated (e.g. "Human keratinocytes") or "not stated".

strain: Strain or stock if given; else null.

sex: "male", "female", "both", or "not stated" (in vivo); use "not applicable" for most ex vivo / in vitro if no animal sex applies.

sample_size: Number of **animals**, **corneas**, **explants**, **cultures**, or wells — number if known, else string estimate, else "not stated".

administration_route: e.g. p.o., topical, instillation, perfusion, immersion, or "not stated".

duration: Timeframe (e.g. "72 hours", "24 h perfusion") or "not stated".

organs_or_tissues: JSON **array of strings** — organs, tissues, or culture systems (e.g. "cornea", "RHE", "liver slice"). Use [] if none.

endpoints_measured: JSON **array of strings** (or a single string) — e.g. LD50, fluorescein retention, TEER, IL-8, histology scores.

test_article: **Required attention:** capture the **substance or product under test** (e.g. "hyaluronate citrate drops") even when the model is ex vivo / in vitro; null only if truly absent.

is_regulatory_standard: Boolean — true if OECD, GLP, ISO, pharmacopoeia, or a named guideline applies to this test.

## Processing rules

- If a section describes **multiple** tests (e.g. skin + eye, or in vivo + ex vivo), create **separate** objects in "experiments".
- **Include ex vivo studies** (slaughterhouse tissues, isolated corneas, perfusion chambers, explants, organ-on-a-chip with animal tissue) as distinct experiments; label **species** as **"Ex vivo [Species]"** when tissue comes from post-mortem / slaughterhouse sources.
- **Expanded trigger keywords — Eye:** conjunctival, corneal, iris, instillation, lacrimation, **EVEIT**, **perfusion**, **isolated cornea**, **explant**, **slaughterhouse**, **ACTO**, **fluorescein**, Draize, BCOP.
- **Skin:** erythema, edema, shaved, patch, corrosion, Draize, **explants**, **reconstructed**, **RhE**, **fibroblasts**.
- **Carcinogenicity:** neoplasm, adenoma, tumor, chronic 2-year, malignancy.
- **Ignore** purely **chemical analysis**, solvent **extraction** methods, plant **extraction** for characterization only, or **statistics-only** sections — do **not** add them as experiments unless they describe a toxicity bioassay.
- If **no** qualifying experiments are found, use an empty array for "experiments" and explain in **study_summary** that no in vivo / ex vivo / in vitro toxicity protocols were identified.

Input Text:
{user_input}'''


def _api_key() -> str:
    return (os.environ.get("API_KEY") or os.environ.get("OPENROUTER_API_KEY") or "").strip()


def mock_response(_prompt: str) -> str:
    return _MOCK_OUTPUT_PATH.read_text(encoding="utf-8")


def complete_openrouter_extraction(prompt: str, *, model: str = DEFAULT_MODEL) -> str:
    user_content = PROTOCOL_EXTRACTOR_PROMPT_TEMPLATE.replace("{user_input}", prompt)

    api_key = _api_key()
    if not api_key:
        raise RuntimeError(
            "API_KEY is not set. Put API_KEY=your_openrouter_key in a .env file in the "
            "project root (next to main.py), or set the environment variable, then restart "
            "the Flask server. You can also use OPENROUTER_API_KEY instead of API_KEY."
        )

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            },
            timeout=120,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not reach OpenRouter: {exc}") from exc

    try:
        data = response.json()
    except ValueError:
        data = {}

    if not response.ok:
        err = data.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or str(err)
        elif isinstance(err, str):
            msg = err
        else:
            msg = (response.text or "").strip() or f"HTTP {response.status_code}"
        raise RuntimeError(msg[:4000])

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        snippet = str(data)[:1500]
        raise RuntimeError(f"Unexpected response shape from OpenRouter: {snippet}")
