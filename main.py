from fastapi import FastAPI,UploadFile,File
import shutil
import os
from inference import predict_emotion
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR="uploads"
os.makedirs(UPLOAD_DIR,exist_ok=True)

@app.get("/")
def root():
    return {"message":"Speech Emotion Recognition API is running "}

@app.post("/predict")
async def predict(file:UploadFile=File(...)):
    print("Received File :",file.filename)
    file_path=os.path.join(UPLOAD_DIR,file.filename)

    with open(file_path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)

    result=predict_emotion(file_path)
    return result