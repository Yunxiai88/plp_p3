import os
import json
import shutil
import logging
import numpy as np
import pandas as pd
from datetime import date

from sentiment import save_mcc, generate_summary
from preprocessing import filter_non
from topic_modelling import TopicModeling

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
        # step1: sentiment
        predictions, _ = sentiment.analyze([review])
        pred = np.argmax(predictions[0], axis=1).flatten()

        if pred == 1:
            polarity = "Positive"
        else:
            polarity = "Negtive"
        print("polarity = " + polarity)

        # step2: topic

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

        # Step 1 : Process Sentiment Analyze
        predictions, true_labels = [], []
        if "Sentiment" in df.columns:
            predictions, true_labels = sentiment.analyze(df["Review"].values, df["Label"].values)
        else:
            predictions, true_labels = sentiment.analyze(df["Review"].values, None)

        # copy MCC to web folder
        if len(true_labels):
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

        # Step 2: Generate Summary for sentiment
        generate_summary()

        # Step 3: Process Topic Modelling
        # positive topic
        positiveTopic = TopicModeling(new_df, 
                                      sentiment_type="Positive", 
                                      sentiment_column_name="Predicted", 
                                      review_column_name="Review")
        # prepressing for topic
        data_lemmatized = positiveTopic.pre_processing()
        lda_model, corpus = positiveTopic.topic_model(2, data_lemmatized)

        # save image
        positive_topic = os.path.join(str(path.parent.parent)+"/webapp/data/img/", "positive.jpg")
        positiveTopic.plot_word_of_importance_chart(lda_model, data_lemmatized, positive_topic)

        #negative topic
        negativeTopic = TopicModeling(new_df, 
                                      sentiment_type="Negative", 
                                      sentiment_column_name="Predicted",
                                      review_column_name="Review")
        # prepressing for topic
        data_lemmatized = positiveTopic.pre_processing()
        lda_model, corpus = positiveTopic.topic_model(4, data_lemmatized)

        # save image
        negative_topic = os.path.join(str(path.parent.parent)+"/webapp/data/img/", "negative.jpg")
        positiveTopic.plot_word_of_importance_chart(lda_model, data_lemmatized, negative_topic)

        return "success"
    except Exception as e:
        print(e)
        return "fail" 

def copy_dataFrame(df):
    columns = ["App", "Review"]
    new_df = df[columns].copy()

    '''
    if "Topic" in df.columns:
        new_df["Topic"] = df["Topic"]
    else:
        new_df.insert(2, "Topic", '', True)
    '''
    
    if "Sentiment" in df.columns:
        new_df["Sentiment"] = df["Sentiment"]
    else:
        new_df.insert(3, "Sentiment", '', True)
    
    return new_df