import os
import json
import shutil
import logging
import numpy as np
import pandas as pd
from datetime import date

logger = logging.getLogger('inference')

def process_review(sentiment, review):
    if review is None: return
    print("single review = " + review)

    polarity = ""
    # process single review
    try:
        predictions, _ = sentiment.analyze([review], None)

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

def batch_process(filename):

    
    # copy image to web folder
        #shutil.copy(fileName, 'webapp/data/img/' + os.path.basename(fileName))
    # return json.dumps(json_result, indent=2)
    return "json_result"

def clean_file(file_path, date):
    for fname in os.listdir(file_path):
        fpath = os.path.join(file_path, fname)

        try:
            if os.path.isfile(fpath):
                # check file name
                if date not in fname:
                    continue

                # delete file
                os.unlink(fpath)

                # log file
                logger.info(fpath + " been removed.")
            elif os.path.isdir(fpath):
                clean_file(fpath, date)
        except Exception as e:
            logger.info('Failed to delete %s. Reason: %s' % (fpath, e))
