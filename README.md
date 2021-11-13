<<<<<<< HEAD
# plp_p3

Google Play Store data which consist of user review and apps Info![image](https://user-images.githubusercontent.com/22022642/138711897-7eaea1ce-38b6-460e-b7c2-1be715db1665.png)
=======
# Block Progress Video Analytics

Keppel Corporation is a Singaporean company that consists of business units in Offshore & Marine, Property, Infrastructure, and Investments. This project aims to leverage on this new monitoring system CCTV’s past block contraction videos, to train a video analytics model to predict future construction progress.

# Technology
* Python: 3.7
* Flask
* OpenCV
* MaskRCNN
* libraries like tensorflow, pandas, numpy

# Training Details
Please refer to [Training on MS COCO](https://github.com/Yunxiai88/bpva_maskrcnn/tree/main/training/maskrcnn)


## Configuration
* 1. create a virtual environment:   
    ```
    conda create -n keppel python=3.7
    ```

* 2. activate newly created environment:   
    ```
    conda activate keppel
    ```

* 3. In the virtual environment, Go to the project root folder and run below command to install packages:   
    ```
    pip install -r requirements.txt  
    ```

     If any packages fail to install, try installing individually      
     If any errors, try to do this one more time to avoid packages being missed out   


## Inference
Navigate to the root of the directory   
Run the detection process for a given day   
Date format to be yyyy_mm_dd, if ommitted, it will process the current day   
eg   
```python
python inference/main.py --date 2020_03_21
```
By default the input videos and the extracxted images will be cleaned after processing   
To disable cleaning, use   
```
python inference/main.py --date 2020_03_21 --clean False  
```

## Start web application
```python
python webapp/webApp.py --ip 127.0.0.1 --port 8000
```
## References
* [How to train Mask R-CNN on the custom dataset. (2021, Apr 21).](https://thebinarynotes.com/how-to-train-mask-r-cnn-on-the-custom-dataset/)
* [Mask R-CNN Demo. (2021, Apr 21).](https://notebook.community/mlperf/community/object_detection/tensorflow/samples/demo)

>>>>>>> add ui
