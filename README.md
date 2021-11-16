# plp_p3

Google Play Store data which consist of user review and apps Info![image](https://user-images.githubusercontent.com/22022642/138711897-7eaea1ce-38b6-460e-b7c2-1be715db1665.png)

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

* 3. install dependent resources:   
    ```
    python -m nltk.downloader stopwords
    python -m spacy download en_core_web_sm
    ```

* 4. In the virtual environment, Go to the project root folder and run below command to install packages:   
    ```
    pip install -r requirements.txt  
    ```

     If any packages fail to install, try installing individually      
     If any errors, try to do this one more time to avoid packages being missed out   


## Inference
TODO

## Start web application
```python
python webapp/webApp.py --ip 127.0.0.1 --port 8000
```
## References
* [BERT Fine-Tuning Tutorial with PyTorch. (2021, Oct 12).](https://mccormickml.com/2019/07/22/BERT-fine-tuning/#a1-saving--loading-fine-tuned-model)
* [Fine-tuning a BERT model. (2021, Oct 12).](https://www.tensorflow.org/text/tutorials/fine_tune_bert)
