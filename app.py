from fastapi import FastAPI, HTTPException, Body, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import json
from datetime import datetime
import base64
from io import BytesIO


app = FastAPI(
    title="API de Análise Regulatória",
    description="API para análise e previsão de mudanças em textos regulatórios",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {"message": "API ok!", "version": "1.0.0"}

uvicorn.run(app, host="0.0.0.0", port=8000)