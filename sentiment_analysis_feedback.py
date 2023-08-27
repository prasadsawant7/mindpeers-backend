from simpletransformers.classification import ClassificationModel, ClassificationArgs
from sklearn.metrics import accuracy_score
import torch
import pandas as pd
import emoji
import os
from data_cleaning import remove_links, remove_special_characters, remove_newline_characters, remove_whitespaces_from_beg_and_end
import asyncio

cuda_available = torch.cuda.is_available()

sa_feedback_model_path = os.path.join(os.path.dirname(__file__), "outputs_sa_feedback")

sa_feedback_model_args = ClassificationArgs()
sa_feedback_model_args.num_train_epochs = 10
sa_feedback_model_args.learning_rate = 1e-5
sa_feedback_model_args.labels_list = [0, 1, 2]
sa_feedback_model_args.do_lower_case = True
sa_feedback_model_args.save_model_every_epoch = False
sa_feedback_model_args.save_best_model = True
sa_feedback_model_args.overwrite_output_dir = True
sa_feedback_model_args.output_dir = sa_feedback_model_path

sa_feedback_model = ClassificationModel(
    model_type="bertweet",
    model_name=sa_feedback_model_path,
    num_labels=3,
    args=sa_feedback_model_args,
    use_cuda=cuda_available,
)

async def retrain_model(text: list[str], labels: list[int]) -> bool:
    print("Cleaning data...")
    for i in range(len(text)):
        text[i] = remove_links(text[i])
        text[i] = remove_special_characters(text[i])
        text[i] = remove_newline_characters(text[i])
        text[i] = remove_whitespaces_from_beg_and_end(text[i])

    await asyncio.sleep(2)

    train_df = pd.DataFrame({
        "text": text,
        "labels": labels
    })

    print("Training model...")
    await asyncio.sleep(2)
    results = sa_feedback_model.train_model(train_df=train_df)

    if results:
        return True

    return False

def get_sentiment_prediction_from_feedback_model(text: str) -> int:
    print("Cleaning data...")
    text = remove_links(text)
    text = remove_special_characters(text)
    text = remove_newline_characters(text)
    text = remove_whitespaces_from_beg_and_end(text)

    print("Predicting Sentiment...")
    predictions, _ = sa_feedback_model.predict([text])

    return predictions[0]
