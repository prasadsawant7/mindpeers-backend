from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentiment_analysis_pretrained import get_sentiment
from keyword_extraction_pretrained import get_keywords

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/api/v1/test")
def index():
    return { "message": "Connection Successfull!" }

@app.post("/api/v1/sentiment-analysis")
async def send_text_to_sentiment_analysis(data: dict) -> dict:
    sentiment = get_sentiment(data["text"])
    response = {
        "text": data["text"],
        "sentiment": sentiment
    }
    return { "status": "ok", "response": response }

@app.post("/api/v1/sentiment-analysis-feedback")
async def send_feedback_to_retrain_sentiment_analysis_model():
    # Storing feedbacks in DB
    return { "status": "ok", "response": "done" }

@app.post("/api/v1/retrain-sentiment-analysis")
async def sentiment_analysis_retraining():
    # Send feedbacks from DB to retrain the model
    return { "status": "ok", "response": "done" }
    
@app.post("/api/v1/keyword-extraction")
async def send_text_to_keyword_extraction(data: dict) -> dict:
    keywords = get_keywords(data["text"])
    response = {
        "text": data["text"],
        "keywords": keywords
    }
    return { "status": "ok", "response": response }

@app.post("/api/v1/keyword-extraction-feedback")
async def send_feedback_to_retrain_keyword_extraction_model():
    # Storing feedbacks in DB
    return { "status": "ok", "response": "done" }

@app.post("/api/v1/retrain-keyword-extraction")
async def keyword_extraction_retraining():
    # Send feedbacks from DB to retrain the model
    return { "status": "ok", "response": "done" }

@app.post("/api/v1/data-tagging")
async def send_text_to_data_tagging():
    # Returns prediction from pretrained model
    return { "status": "ok", "response": "done" }

@app.post("/api/v1/data-tagging-feedback")
async def send_feedback_to_retrain_data_tagging_model():
    # Storing feedbacks in DB
    return { "status": "ok", "response": "done" }

@app.post("/api/v1/retrain-data-tagging")
async def data_tagging_retraining():
    # Send feedbacks from DB to retrain the model
    return { "status": "ok", "response": "done" }