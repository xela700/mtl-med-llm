"""
API setup for demoing model pipeline

Uses FastAPI to create a simple web interface for sending text to the model
and receiving output.
"""

import torch
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
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

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <body>
        <h1>Biomedical Multi-Model LLM Demo</h1>
        <textarea id="input" rows="8" cols="80"></textarea><br>
        <button onclick="sendRequest()">Run Model</button>
        <h3>Output:</h3>
        <pre id="output"></pre>

    <script>
        async function sendRequest() {
            const input = document.getElementById('input').value;
            const response = await fetch('/infer', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: input})
            });
            const data = await response.json();
            document.getElementById('output').textContent += 
                JSON.stringify(data) + "\\n\\n"; // <-- append, don't overwrite
        }
    </script>
    </body>
    </html>
    """