from langchain_ollama import OllamaLLM
from app.core.config import settings
from app.rag.retriever import retrieve_context

llm = OllamaLLM(
    model=settings.ollama_model,
    base_url=settings.ollama_base_url
)

SYSTEM_PROMPT = """You are a compassionate medical assistant specializing 
in diabetes. Explain the risk assessment in plain English. Cover:
1. What the key values mean
2. Why this risk level was assigned  
3. Three actionable lifestyle suggestions
Keep it under 150 words. Never say this is a diagnosis."""

def explain_prediction(features: dict, ml_result: dict) -> str:
    summary = f"""
Patient values:
- Glucose: {features['Glucose']} mg/dL
- BMI: {features['BMI']}
- Age: {features['Age']}
- Blood Pressure: {features['BloodPressure']} mm Hg
- Insulin: {features['Insulin']} mu U/ml
- Pregnancies: {features['Pregnancies']}
- Skin Thickness: {features['SkinThickness']} mm
- Diabetes Pedigree: {features['DiabetesPedigreeFunction']}

ML Result: {ml_result['label']}
Confidence: {ml_result['probability'] * 100:.1f}%
Risk level: {ml_result['risk_level']}
Key factors: {', '.join(ml_result['top_factors'])}
"""

    context = retrieve_context(summary)

    full_prompt = f"""{SYSTEM_PROMPT}

Patient Summary:
{summary}

Relevant medical knowledge:
{context}

Provide your assessment:"""

    return llm.invoke(full_prompt)