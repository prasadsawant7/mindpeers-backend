from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase_db import create_text_labels, read_text_labels, update_text_labels, delete_text_labels
from sentiment_analysis_pretrained import get_sentiment_prediction
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
    prediction: int = get_sentiment_prediction(data["text"])
    if prediction:
        if prediction == 0:
            sentiment = "Negative"
        if prediction == 1:
            sentiment = "Neutral"
        if prediction == 2:
            sentiment = "Positive"
        response = {
            "text": data["text"],
            "sentiment": sentiment
        }
        return { "status_code": 200, "response": response }
    return { "status_code": 500, "response": "Internal Server Error" }

@app.post("/api/v1/sentiment-analysis-feedback")
async def send_feedback_to_retrain_sentiment_analysis_model(feedback: dict) -> dict:
    res = create_text_labels(feedback)
    if res:
        response = {
            "id": res["id"],
            "text": res["text"],
            "sentiment": res["labels"],
            "is_trained": res["is_trained"]
        }
        return { "status_code": 201, "response": response }
    return { "status_code": 500, "response": "Internal Server Error" }

@app.get("/api/v1/sentiment-analysis-feedback")
async def get_untrained_feedback() -> dict:
    res = read_text_labels()
    if res:
        return { "status_code": 200, "response": res }
    return { "status_code": 204, "response": "No untrained feedbacks" }

@app.put("/api/v1/retrain-sentiment-analysis")
async def sentiment_analysis_retraining() -> dict:
    # Fetching untrained feedbacks from DB
    untrained_feedbacks = read_text_labels()
    if len(untrained_feedbacks) >= 10:
        # Model will train here
        text = [feedback["text"] for feedback in untrained_feedbacks]
        labels = [feedback["labels"] for feedback in untrained_feedbacks]

        # After training model update untrained feedbacks status to trained feedbacks
        response = []
        for feedback in untrained_feedbacks:
            data = update_text_labels(feedback["id"])
            response.append({
                "id": data["id"],
                "text": data["text"],
                "sentiment": data["labels"],
                "is_trained": data["is_trained"]
            })
        return { "status_code": 200, "response": response }
    return { "status_code": 422, "response": "Feedback are less to train the model, give 10 or more than 10 feedbacks to train the model" }

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