import os
import argparse
import numpy as np

def filter_non(data):
    data = data[data.Review.notnull()]

    if 'Sentiment' in data.columns:
        data = data[data.Sentiment.notnull()]
        data["Label"] = data["Sentiment"].apply(lambda l: int(l == 'Positive'))

    return data

# remove words which length less than 2
def preprocess(sentence) :
    words_clean = [w.lower() for w in sentence.split() if len(w) >= 3]
    words_clean = [ w for w in words_clean if not w.isdigit() ]
    return words_clean