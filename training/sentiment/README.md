# Sentiment Analysis for ***

BERT (Bidirectional Encoder Representations from Transformers), released in late 2018, is the model we will use in this tutorial to provide readers with a better understanding of and practical guidance for using transfer learning models in NLP.
this file will show you how to use BERT with the huggingface PyTorch library to quickly and efficiently fine-tune a model to get near state of the art performance in sentence classification.

# Training
We use pre-trained weights as a starting point to train our own annotationed dataset.

# Train a new model starting from pre-trained BERT model (BertForSequenceClassification)
Pls refer to Sentiment-Mining.ipynb for more details

## Requirements
Python 3.7, torch, transformers and other common packages listed in `requirements.txt`.

## Installation
1. Clone this repository
2. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```

# Getting Started
* [Sentiment-Mining.ipynb](training/sentiment/Sentiment-Mining.ipynb). This notebook visualizes the different pre-processing steps
to prepare the training data and how to train the model step by steps.
