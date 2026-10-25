# Deepfake Image Classification

A Flask application that applies a saved deep convolutional neural network to one uploaded face image. The notebook trains a grayscale 128 x 128 CNN with convolution, pooling, and global-average-pooling layers, followed by a sigmoid binary-class output. It uses Adam and binary cross-entropy and records ROC AUC and average precision during evaluation.

The notebook reads the real/fake image folders in the `real-vs-fake` dataset tree and builds train, validation, and test generators. The inference path preserves the recorded grayscale 128 x 128 preprocessing. `app/deepfake.h5` is the saved classifier, `ML_model/DFD.ipynb` records the experiment, and `tests/` covers upload and preprocessing behavior. PNG/JPEG uploads are processed in memory with an 8 MB request limit.

## Run locally

```sh
python -m venv .venv
# Activate .venv, then:
python -m pip install -r requirements.txt
python app/app.py
```
