from simpletransformers.classification import ClassificationModel, ClassificationArgs
import torch

cuda_available = torch.cuda.is_available()

dt_model = ClassificationModel(
    model_type="bert",
    model_name="Modifly/Mental-Health-Classification",
    use_cuda=cuda_available
)

preds, _ = dt_model.predict(["I love you"])

print(preds)

# def get_sentiment(text: str) -> list:
#     prediction, _ = sa_model.predict([text])
#     sentiment = None
#     if prediction[0] == 0:
#         sentiment = "Negative"
#     if prediction[0] == 1:
#         sentiment = "Neutral"
#     if prediction[0] == 2:
#         sentiment = "Positive"
#     return sentiment