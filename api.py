"""
API setup for demoing model pipeline
"""

import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from model.model_inference import model_routing_pipeline

app = FastAPI(title="Biomedical Modular LLM")

class InferenceRequest(BaseModel):
    text: str

@app.post("/infer")
def infer(request: InferenceRequest):
    result = model_routing_pipeline(request.text)

    if result is None:
        return {"error": "No model proeduced output.", "input": request.text}

    if isinstance(result, (torch.Tensor, np.ndarray)):
        result = result.tolist()

    return {"result": result}
