# Deepfake Image Classification Demo

A Flask demo that applies a saved Keras classifier to one uploaded face image. It is an image-level experiment, not a forensic authenticity system or video detector.

## What is included

- PNG/JPEG uploads handled in memory with an 8 MB limit
- Raw-pixel grayscale 128 × 128 preprocessing to match the recorded training contract
- Regression tests for upload and preprocessing behavior

## Run locally

```sh
python -m venv .venv
# Activate .venv, then:
python -m pip install -r requirements.txt
python app/app.py
```

## Verify

python -m pip install -r requirements-dev.txt
python -m pytest -q

## Repository map

`app/` contains the Flask interface and inference path. `app/deepfake.h5` is the saved model; `ML_model/DFD.ipynb` is the experiment notebook. `tests/` covers request/preprocessing behavior.

## Evidence and limitations

The supplied model is wired into the upload path. Automated tests cover interface contracts, not cross-source detection accuracy or evidentiary reliability.

A prediction is not proof that media was manipulated. This project does not analyze video or provide forensic provenance. Dataset provenance, license, independent cross-dataset performance and subgroup behavior remain unverified. Do not use for consequential decisions.

## License

No repository license file is included. Unless the project owner adds a license, do not assume the code, data or model artifacts may be redistributed or reused. Verify third-party data and model terms separately.
