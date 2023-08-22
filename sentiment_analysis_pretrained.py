from simpletransformers.classification import ClassificationModel, ClassificationArgs
import torch
import emoji
from data_cleaning import remove_links, remove_special_characters, remove_newline_characters, remove_whitespaces_from_beg_and_end

cuda_available = torch.cuda.is_available()

sa_model = ClassificationModel(
    model_type="bertweet",
    model_name="finiteautomata/bertweet-base-sentiment-analysis",
    use_cuda=cuda_available
)

def get_sentiment_prediction(text: str) -> int:
    print("Cleaning data...")
    text = remove_links(text)
    text = remove_special_characters(text)
    text = remove_newline_characters(text)
    text = remove_whitespaces_from_beg_and_end(text)

    print("Predicting Sentiment...")
    predictions, _ = sa_model.predict([text])

    return int(predictions[0])