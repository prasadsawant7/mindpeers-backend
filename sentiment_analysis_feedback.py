from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import accuracy_score
import torch
import pandas as pd
import os

cuda_available = torch.cuda.is_available()

model_path = os.path.join(os.path.dirname(__file__), "outputs_feedback")

model_args = ClassificationArgs()
model_args.num_train_epochs = 10
model_args.learning_rate = 1e-4
model_args.labels_list = [0, 1]
model_args.do_lower_case = True
model_args.output_dir = model_path
model_args.overwrite_output_dir = True
model_args.save_model_every_epoch = False
model_args.save_best_model = True

model = ClassificationModel('bert', model_path, num_labels=2, args=model_args, use_cuda=cuda_available)

def send_sa_feedback(text: str, label: int):
    train_df = pd.DataFrame({
        "text": [text],
        "label": [label]
    })
    model.train_model(train_df=train_df, acc=accuracy_score)
    predictions, _ = model.predict([text])
    return predictions[0]

def get_sentiment_after_feedback(text: str) -> str:
    predictions, _ = model.predict([text])
    sentiment = "positive" if predictions[0] == 1 else "negative"
    return sentiment