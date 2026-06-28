import os
import sys
import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(str(Path(__file__).parent))
sys.path.append('/app/src')
from agents import build_graph, AgentState

app = FastAPI(
    title="AI Face Detector API",
    description="Detects whether an image is AI generated or real",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


pipeline = build_graph()


TEMP_DIR = Path("temp_uploads")
TEMP_DIR.mkdir(exist_ok=True)



class AnalysisResult(BaseModel):
    verdict: str
    confidence: float
    is_ai: bool
    quality_warning: str
    quality_passed: bool
    texture_score: float
    edge_score: float
    color_score: float
    explanation: str



@app.get("/health")
def health():
    return {"status": "ok"}



@app.post("/analyze", response_model=AnalysisResult)
async def analyze_image(file: UploadFile = File(...)):
    """
    Accepts an image file, runs it through the LangGraph pipeline,
    returns verdict, confidence, explanation.
    """
    
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(
            status_code=400,
            detail="Only JPEG and PNG images are supported"
        )

    
    temp_path = TEMP_DIR / f"{uuid.uuid4()}.jpg"
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        
        initial_state = AgentState(
            image_path=str(temp_path),
            image_pil=None,
            quality_passed=False,
            quality_warning="",
            is_ai=False,
            confidence=0.0,
            texture_score=0.0,
            edge_score=0.0,
            color_score=0.0,
            verdict="",
            explanation=""
        )

        
        result = pipeline.invoke(initial_state)

        return AnalysisResult(
            verdict=result['verdict'],
            confidence=result['confidence'],
            is_ai=result['is_ai'],
            quality_warning=result['quality_warning'],
            quality_passed=result['quality_passed'],
            texture_score=result['texture_score'],
            edge_score=result['edge_score'],
            color_score=result['color_score'],
            explanation=result['explanation']
        )

    finally:
        
        if temp_path.exists():
            temp_path.unlink()



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)