# Credit Card Fraud Detection (work in progress)

*Created as a learning project within the 
[Clark Summer Research Program](https://honors.utdallas.edu/clark-summer-research-program)
at the University of Texas at Dallas.*

**Disclaimer: This is by no means efficient (or perfectly working) code. Please don't use in production.**

## About this Model
This model is a single-layer neural network, with the architecture as follows:

```text
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 5)                 150       
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 6         
=================================================================
Total params: 156
Trainable params: 156
Non-trainable params: 0
_________________________________________________________________

```


The first `Dense` layer has a ReLU activation function, and the output layer has a `Sigmoid` activation function. 

## Performance
This model performs with 99.99% accuracy on the test set, as generated through a deterministic random shuffle of the original dataset.

This model also does relatively well when evaluated on other metrics, such as F₁ score (0.77-0.80 on the test set). 
A higher score is not practically achievable due to the imbalance between positive and negative examples within the dataset (there are much fewer fraud transactions vs. non-fraud).

## Install
1. `virtualenv env; source env/bin/activate`
2. `pip install -r requirements.txt`
3. Put Credit Card Fraud dataset under `data/train.csv`
4. `python main.py`

## Pretrained Model
A pretrained model is located under `models/`. Its statistics on the test set are contained in its file name.

## Data Ownership
This dataset (Credit Card Fraud) is licensed under the Database Contents License and is used under its terms.
The originator of the dataset is the ULB Machine Learning Group. 