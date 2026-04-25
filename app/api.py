from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.config import MODEL_BUNDLE_PATH
from pricing.clo_pricing import predict_from_bundle

app = FastAPI(title="CLO Tranche Pricing API", version="0.3.0")


class NewPayloadRequest(BaseModel):
    payload: dict[str, Any]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict/new")
def predict_new(request: NewPayloadRequest):
    try:
        df = pd.DataFrame([request.payload])
        return predict_from_bundle(df, str(MODEL_BUNDLE_PATH))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    try:
        temp_path = Path("/tmp") / str(file.filename)
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        if temp_path.suffix.lower() == ".csv":
            df = pd.read_csv(temp_path)
        else:
            df = pd.read_excel(temp_path)

        return predict_from_bundle(df.head(1), str(MODEL_BUNDLE_PATH))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
