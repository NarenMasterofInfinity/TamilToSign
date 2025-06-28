from fastapi import FastAPI, UploadFile, File, Form
from typing import Optional

app = FastAPI()

@app.post("/stt_audio")
async def stt_audio(audio: UploadFile = File(...)):
    # ...convert audio to text...
    return None

@app.post("/stt_video")
async def stt_video(video: UploadFile = File(...)):
    # ...convert video to audio to text...
    return None

@app.post("/text_to_gloss")
async def text_to_gloss(text: str = Form(...)):
    # ...convert text to gloss...
    return None

@app.post("/gloss_to_pose")
async def gloss_to_pose(gloss: str = Form(...)):
    # ...convert gloss to pose...
    return None

@app.post("/gloss_to_video")
async def gloss_to_video(gloss: str = Form(...)):
    # ...convert gloss to video...
    return None

@app.get("/retrieve_history")
async def retrieve_history():
    # ...get video and text...
    return None

@app.post("/store_data")
async def store_data(text: str = Form(...), video_path: str = Form(...)):
    # ...store video and text in history...
    return None
