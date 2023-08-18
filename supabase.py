from dotenv import load_dotenv
load_dotenv()

import os
from supabase import create_client, Client

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(url, key)

def create_text_labels(text: str, labels: int) -> dict:
    data = supabase.table("sentiment_analysis_feedback").insert({"text": text, "labels": labels}).execute().data[0]
    return data

def read_text_labels() -> list:
    data = supabase.table("sentiment_analysis_feedback").select("id, text, labels").eq("is_trained", False).execute().data
    return data

def update_text_labels(id: int) -> dict:
    data = supabase.table("sentiment_analysis_feedback").update({"is_trained": True}).eq("id", id).execute().data[0]
    return data

def delete_text_labels(id: int) -> dict:
    data = supabase.table("sentiment_analysis_feedback").delete().eq("id", id).execute().data[0]
    return data