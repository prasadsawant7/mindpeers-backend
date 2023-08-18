from keybert import KeyBERT
from nltk.corpus import stopwords
import nltk

nltk.download("stopwords")
stopwords_list = stopwords.words("english")

ke_model = KeyBERT(model="yanekyuk/bert-uncased-keyword-extractor")

def get_keywords(text: str) -> list:
    keywords = ke_model.extract_keywords(
        text,
        stop_words=stopwords_list,
        keyphrase_ngram_range=(1, 1),
        top_n=len(text.split(" ")) * 40 // 100
    )
    relevant_keywords = [(keyword, score) for keyword, score in keywords if score >= 0.4]
    return relevant_keywords