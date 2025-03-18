from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import utils
import os
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

app = FastAPI(title="News Sentiment Analysis API",
              description="API for analyzing sentiment of news articles related to a company",
              version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CompanyRequest(BaseModel):
    company_name: str
    num_articles: Optional[int] = 10

@app.post("/analyze")
async def analyze_company_news(request: CompanyRequest):
    """
    Analyze news articles for a given company.
    
    Args:
        request: Company request object
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Process company news
        result = utils.process_company_news(request.company_name, request.num_articles)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing news: {str(e)}")

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """
    Retrieve an audio file.
    
    Args:
        filename: Audio filename
        
    Returns:
        Audio file
    """
    file_path = f"data/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(file_path, media_type="audio/mp3")

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Run the FastAPI app
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
