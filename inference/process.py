import os
import json
import shutil
import logging
import numpy as np
import pandas as pd
from datetime import date

from sentiment import save_mcc
from preprocessing import filter_non

#append system path
import pathlib
path = pathlib.Path(__file__)

logger = logging.getLogger('inference')

def single_process(sentiment, review):
    if review is None: return
    print("single review = " + review)

    polarity = ""
    # process single review
    try:
        predictions, _ = sentiment.analyze([review])

        pred = np.argmax(predictions[0], axis=1).flatten()

        if pred == 1:
            polarity = "Positive"
        else:
            polarity = "Negtive"
        print("polarity = " + polarity)
    except Exception as e:
        print(e)

    # return json with correct result
    return polarity

def batch_process(sentiment, filename):
    if filename is None: return
    print("batch process job start..")

    try:
        data_file = os.path.join(str(path.parent.parent)+"/upload/", filename)
        df = pd.read_csv(data_file)

        # Filter None
        df = filter_non(df)

        # Sentiment Analyze
        predictions, true_labels = [], []
        if "Sentiment" in df.columns:
            predictions, true_labels = sentiment.analyze(df["Review"].values, df["Label"].values)
        else:
            predictions, true_labels = sentiment.analyze(df["Review"].values, None)

        # copy MCC to web folder
        if true_labels:
            save_mcc(predictions, true_labels)
        
        # Save to webfolder
        flat_predictions = np.concatenate(predictions, axis=0)
        flat_pred_labels = np.argmax(flat_predictions, axis=1).flatten()
        flat_pred_labels = ['Positive' if l == 1 else 'Negitive' for l in flat_pred_labels]

        # Copy to new DataFrame
        new_df = copy_dataFrame(df)
        new_df['Predicted'] = flat_pred_labels

        output_file = os.path.join(str(path.parent.parent)+"/webapp/data/", "progress.csv")
        new_df.to_csv(output_file, index=False)

        return "success"
    except Exception as e:
        print(e)
        return "fail" 

def copy_dataFrame(df):
    columns = ["App", "Review"]
    new_df = df[columns].copy()

    if "Topic" in df.columns:
        new_df["Topic"] = df["Topic"]
    else:
        new_df.insert(2, "Topic", '', True)
    
    if "Sentiment" in df.columns:
        new_df["Sentiment"] = df["Sentiment"]
    else:
        new_df.insert(3, "Sentiment", '', True)
    
    return new_df