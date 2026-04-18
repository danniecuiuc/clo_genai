from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from common.config import MODEL_BUNDLE_PATH
from common.io import load_dataframe
from product_pipeline.predict import predict_for_existing_id, predict_for_new_payload

app = FastAPI(title='CLO Tranche Pricing API', version='0.2.0')


class ExistingIdRequest(BaseModel):
    bloomberg_id: str
    k: int = 10


class NewPayloadRequest(BaseModel):
    payload: dict[str, Any]
    k: int = 10


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/predict/by-id')
def predict_by_id(request: ExistingIdRequest):
    try:
        result = predict_for_existing_id(request.bloomberg_id, str(MODEL_BUNDLE_PATH), request.k)
        return result.__dict__
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post('/predict/new')
def predict_new(request: NewPayloadRequest):
    try:
        df = pd.DataFrame([request.payload])
        result = predict_for_new_payload(df, str(MODEL_BUNDLE_PATH), request.k)
        return result.__dict__
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post('/predict/upload')
async def predict_upload(file: UploadFile = File(...), k: int = 10):
    try:
        temp_path = f'/tmp/{file.filename}'
        with open(temp_path, 'wb') as f:
            f.write(await file.read())
        df = load_dataframe(temp_path)
        result = predict_for_new_payload(df.head(1), str(MODEL_BUNDLE_PATH), k)
        return result.__dict__
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
