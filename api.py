"""
API setup for demoing model pipeline
"""

from fastapi import FastAPI
from pydantic import BaseModel
from model.model_inference import model_routing_pipeline

app = FastAPI(title="Biomedical Modular LLM")

class InferenceRequest(BaseModel):
    text: str

@app.post("/infer")
def infer(request: InferenceRequest):
    result = model_routing_pipeline(request.text)
    return {"result": result}
