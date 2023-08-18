from simpletransformers.classification import ClassificationModel, ClassificationArgs
import torch
import emoji

cuda_available = torch.cuda.is_available()

sa_model = ClassificationModel(
    model_type="bertweet",
    model_name="finiteautomata/bertweet-base-sentiment-analysis",
    use_cuda=cuda_available
)

def get_sentiment_prediction(text: str) -> int:
    predictions, _ = sa_model.predict([text])
    return predictions[0]