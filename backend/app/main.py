from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import our inference logic
from app.inference import process_image_frame, initialize_models

app = FastAPI(title="Vehicle Analytics API", version="1.0.0")

# Configure CORS so Streamlit can talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    """Render.com ping endpoint."""
    return {"status": "healthy", "message": "Backend is active."}

@app.post("/detect/frame")
async def detect_frame(
    file: UploadFile = File(...),
    fps: float = Form(30.0) # Allow Streamlit to send the video's actual FPS
):
    """Receives a single frame, processes it, and returns tracking data."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Must be an image file.")
    
    try:
        image_bytes = await file.read()
        
        # Pass bytes and FPS to inference
        result = process_image_frame(image_bytes, video_fps=fps)
        
        return {"success": True, "data": result}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset")
def reset_tracker():
    """Call this from Streamlit whenever a new video starts."""
    try:
        initialize_models()
        return {"success": True, "message": "Tracking history cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)