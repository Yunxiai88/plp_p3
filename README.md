# GOOGLE PLAY STORE SENTIMENT ANALYSIS

Team has prepared to deliver an end-to-end solution for these companies to leverage on. This web-based system allows users to upload a batch file to identify Positive and Negative sentiments from Play store review through sentiment analysis. System also can perform sentiment analysis for single review related to any app in google play store.
This system also supports using topic modelling to understand which topic or feature customers like/dislike under different app categories.


# Technology
* Python: 3.7
* Flask
* Transformer
* BERT
* libraries like torch, pandas, numpy

# Training Details
Please refer to [Training on BERT for Sentiment Analysis](https://github.com/Yunxiai88/plp_p3/tree/main/training/sentiment)

Please refer to [Training for Topic Modelling](https://github.com/Yunxiai88/plp_p3/tree/main/training/topic)

## Configuration
* 1. create a virtual environment:   
    ```
    conda create -n sentiment python=3.7
    ```

* 2. activate newly created environment:   
    ```
    conda activate sentiment
    ```

* 3. In the virtual environment, Go to the project root folder and run below command to install packages:   
    ```
    pip install -r requirements.txt  
    ```

     If any packages fail to install, try installing individually      
     If any errors, try to do this one more time to avoid packages being missed out   

* 4. install dependent resources:   
    ```
    python -m nltk.downloader stopwords
    python -m spacy download en_core_web_sm
    ```

## Inference
TODO

## Start web application
```python
python webapp/webApp.py --ip 127.0.0.1 --port 8000
```
## References
* [BERT Fine-Tuning Tutorial with PyTorch. (2021, Oct 12).](https://mccormickml.com/2019/07/22/BERT-fine-tuning/#a1-saving--loading-fine-tuned-model)
* [Fine-tuning a BERT model. (2021, Oct 12).](https://www.tensorflow.org/text/tutorials/fine_tune_bert)
* [Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
* [Sentiment Analysis Using Neural Network](https://www.kaggle.com/atillasilva/sentiment-analysis-using-neural-network)
* [Amazon Product Review Sentiment Analysis using BERT](https://www.analyticsvidhya.com/blog/2021/06/amazon-product-review-sentiment-analysis-using-bert/)
* [Topic modeling visualization – How to present the results of LDA models](https://www.machinelearningplus.com/nlp/topic-modeling-visualization-how-to-present-results-lda-models/)
* [Text Clustering using K-means](https://towardsdatascience.com/text-clustering-using-k-means-ec19768aae48)
* [Understanding K-means Clustering in Machine Learning](https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1)
