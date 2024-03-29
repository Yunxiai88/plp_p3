import os
import logging
import pandas as pd
import matplotlib as mpl
logger = logging.getLogger('inference')
if os.environ.get('DISPLAY','') == '':
    logger.info('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

import sys
import pathlib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef

import torch
from nltk import FreqDist
from transformers import BertTokenizer, BertForSequenceClassification

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, SequentialSampler

#append system path
path = pathlib.Path(__file__)

class Sentiment:
    def __init__(self):
        self.device = get_device()
        self.model, self.tokenizer = load_model(self.device)

    def analyze(self, data, labels=[]):
        print("sentiment analysis start...")

        # Generate Input Data
        dataset = generate_input(data, labels, self.tokenizer)

        dataloader = DataLoader(
            dataset,                              # The validation samples.
            sampler = SequentialSampler(dataset), # Pull out batches sequentially.
            batch_size = 32                       # Evaluate with this batch size.
        )

        # Run Analyze
        predictions , true_labels = [], []

        step = 1
        for batch in dataloader:
            print("processing for batch: " + str(step))

            batch = tuple(t.to(self.device) for t in batch)
            if len(batch) == 3:
                b_input_ids, b_input_mask, b_labels = batch
            else:
                b_input_ids, b_input_mask = batch

            with torch.no_grad():
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            # Move logits and labels to CPU
            logits = outputs.logits.detach().cpu().numpy()
            predictions.append(logits)

            if len(batch) == 3:
                label_ids = b_labels.to('cpu').numpy()
                true_labels.append(label_ids)
            
            step = step + 1
        
        print("sentiment analysis end...")

        return predictions, true_labels

def load_model(device):
    print("loading model start..")

    output_dir = os.path.join(str(path.parent.parent), "model_save/")
    model = BertForSequenceClassification.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained(output_dir)

    # Copy the model to the GPU.
    model.to(device)

    print("loading model end..")
    return model, tokenizer

def get_device():
    # If there's a GPU available...
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    return device

def generate_input(input_data, input_labels, tokenizer):
  input_ids = []
  attention_masks = []

  for sent in input_data:
      encoded_dict = tokenizer.encode_plus(
                          sent,                      # Sentence to encode.
                          add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                          max_length = 100,          # Pad & truncate all sentences.
                          padding = 'max_length',
                          truncation=True,
                          return_attention_mask = True,   # Construct attn. masks.
                          return_tensors = 'pt',          # Return pytorch tensors.
                    )
      
      # Add the encoded sentence to the list.    
      input_ids.append(encoded_dict['input_ids'])
      
      # And its attention mask (simply differentiates padding from non-padding).
      attention_masks.append(encoded_dict['attention_mask'])

  # convert to tensors
  input_ids = torch.cat(input_ids, dim=0)
  attention_masks = torch.cat(attention_masks, dim=0)
  
  if len(input_labels):
    labels = torch.tensor(input_labels)
    return TensorDataset(input_ids, attention_masks, labels)
  else:
    return TensorDataset(input_ids, attention_masks)

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def total_accuracy(predictions, true_labels):
    total_accuracy = 0
    for p, t in zip(predictions, true_labels):
        print("Accuracy for batch: ", flat_accuracy(p, t))
        total_accuracy += flat_accuracy(p, t)

    print("Overall accuracy for test set:", total_accuracy/len(predictions))

def save_mcc(predictions, true_labels):
    matthews_set = []
    pred_labels = []
    # Evaluate each test batch using Matthew's correlation coefficient
    print('Calculating Matthews Corr. Coef. for each batch...')

    # For each input batch, convert the predicted logits to 0/1 labels...
    for i in range(len(predictions)):
        # The predictions for this batch are a 2-column ndarray (one column for "0" 
        # and one column for "1"). Pick the label with the highest value and turn this
        # in to a list of 0s and 1s.
        pred_labels.append(np.argmax(predictions[i], axis=1).flatten())
    

    # For each input batch...
    for i in range(len(true_labels)):
        # Calculate and store the coef for this batch.  
        matthews = matthews_corrcoef(true_labels[i], pred_labels[i])                
        matthews_set.append(matthews)

    # Save Image
    sns.set(style='darkgrid')
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    plt.title('MCC Score per Batch')
    plt.ylabel('MCC Score (-1 to +1)')
    plt.xlabel('Batch #')

    ax = sns.barplot(x=list(range(len(matthews_set))), y=matthews_set, ci=None)

    mcc_file = str(path.parent.parent) + "/webapp/data/img/sentiment_mcc.jpg"
    plt.savefig(mcc_file)

def generate_summary():
    print("generate summary for sentiment")
    data_file = os.path.join(str(path.parent.parent)+"/webapp/data/", "progress.csv")
    df1 = pd.read_csv(data_file)

    reference_file = os.path.join(str(path.parent.parent)+"/webapp/data/", "googleplaystore.csv")
    df2 = pd.read_csv(reference_file)

    # merger with app store info
    df2 = pd.merge(df1, df2[{"App", "Category", "Installs"}], on=['App'], how='left')

    # top 10 apps
    top10_apps_name = df2.sort_values('Installs', ascending=False)["App"].drop_duplicates().head(10).values

    df = df2.loc[df2['App'].isin(top10_apps_name)]

    sm = df.groupby(['App','Predicted']).agg({'Predicted': 'count'}).groupby(level=0).apply(lambda x : round(100 * x / float(x.sum()), 2))
    #sm.columns = ['sentiment_mean']
    #sm = sm.reset_index()

    # Generate Sentiment Summary
    ax = sm.unstack(fill_value=0).plot(kind='bar', title ="Top 10 Popular Apps", figsize=(15, 10), legend=True, fontsize=12)
    ax.set_xlabel("App Name", fontsize=12)
    ax.set_ylabel("Sentiment Percentage", fontsize=12)

    sentiment = str(path.parent.parent) + "/webapp/data/img/sentiment.jpg"
    plt.savefig(sentiment)

    # Generate Summary for top 10 apps
    '''
    seq = 1
    for app_name in top10_apps_name:
        filename = "sentiment"+str(seq)+".jpg"
        seq = seq + 1

        df_topic = get_topic(sm, app_name)
        
        ## Setting figure, ax into variables
        fig, ax = plt.subplots(figsize=(10,10))

        ## Seaborn plotting using Pandas attributes + xtick rotation for ease of viewing
        sns.barplot(y=df_topic.index, x=df_topic.values, orient='h', ax=ax)
        
        plt.title('Top 10 Topic')
        plt.ylabel('Frequency')
        plt.xlabel('Topic')
        plt.xticks(rotation=30)

        app_topic = str(path.parent.parent) + "/webapp/data/img/" + filename
        plt.savefig(app_topic)
    '''

def get_topic(df, appname):
    df_app = df.loc[df.App == appname]
    df_app = df_app[df_app['Topic'].notnull()]

    words = []
    for x in df_app.Topic.values:
        if x and x != "":
            words = words + [y.strip() for y in x.split(",")]

    fd_req = FreqDist(words).most_common(10)
    fd_req = pd.Series(dict(fd_req))
    return fd_req