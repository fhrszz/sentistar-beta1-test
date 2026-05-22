from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentiment import analisis_sentimen
import traceback

app = FastAPI(title="SentiStar API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # bisa diperketat ke domain GitHub Pages kamu nanti
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class AnalysisRequest(BaseModel):
    teks: str = Field(..., min_length=1, max_length=2000)
    bintang: int = Field(..., ge=1, le=5)


class AnalysisResponse(BaseModel):
    label_model:  str
    conf_model:   float
    bobot_model:  float
    bobot_rating: float
    label_akhir:  str
    skor_akhir:   float
    ada_slang:    bool
    berubah:      bool


@app.get("/")
def root():
    return {"status": "ok", "message": "SentiStar API berjalan"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(req: AnalysisRequest):
    try:
        result = analisis_sentimen(req.teks, req.bintang)
        return AnalysisResponse(**result)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
