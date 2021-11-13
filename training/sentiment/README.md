# Mask R-CNN for Block Progress Video Analysis

This is an implementation of [Mask R-CNN](https://github.com/matterport/Mask_RCNN) on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

# Training on MS COCO
We use pre-trained weights as a starting point to train our own annotationed dataset.
Training and evaluation code is in `samples/keppel/keppel.py`. To run it directly from the command line as such:

```
# Train a new model starting from pre-trained COCO weights
python3 samples/keppel/keppel.py train --dataset=/path/to/keppel/ --model=coco

# Train a new model starting from ImageNet weights
python3 samples/keppel/keppel.py train --dataset=/path/to/keppel/ --model=imagenet

# Continue training a model that you had trained earlier
python3 samples/keppel/keppel.py train --dataset=/path/to/keppel/ --model=/path/to/weights.h5

# Continue training the last model you trained. This will find
# the last trained weights in the model directory.
python3 samples/keppel/keppel.py train --dataset=/path/to/keppel/ --model=last
```

You can also run the COCO evaluation code with:
```
# Run COCO evaluation on the last trained model
python3 samples/keppel/keppel.py evaluate --dataset=/path/to/keppel/ --model=last
```

The training objects, learning rate, and other parameters should be set in `samples/keppel/keppel.py`.
```
* NUM_CLASSES = 1 + 2  # Background + classes of your project
* self.add_class("object", 1, "DK") # object types
* name_dict = {"DK": 1, "ST": 2} # name dictionary
```

## Requirements
Python 3.7, TensorFlow 1.15, Keras 2.0.8 and other common packages listed in `requirements.txt`.

## Installation
1. Clone this repository
2. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```
3. Run setup from the repository root directory
    ```bash
    python3 setup.py install
    ``` 
3. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases).
4. (Optional) If you use Docker, the code has been verified to work on this [Docker container](https://hub.docker.com/r/waleedka/modern-deep-learning/)..

# Getting Started
* [inspect_data.ipynb](samples/keppel/inspect_data.ipynb). This notebook visualizes the different pre-processing steps
to prepare the training data.

* [inspect_model.ipynb](samples/keppel/inspect_model.ipynb) This notebook goes in depth into the steps performed to detect and segment objects. It provides visualizations of every step of the pipeline.

* [inspect_weights.ipynb](samples/keppel/inspect_weights.ipynb)
This notebooks inspects the weights of a trained model and looks for anomalies and odd patterns.
