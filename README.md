# Deploying a Machine Learning Model with FastAPI

Project repository: https://github.com/mirkanodawit-source/Deploying-a-Scalable-ML-Pipeline-with-FastAPI

This project trains a Census income classifier, evaluates held-out categorical
slices, and serves predictions through FastAPI. It follows the supplied Udacity
starter kit and project rubric.

## Set up

Use Python 3.11, the same version configured in GitHub Actions. The requirements
use `requirements-lock.txt` to constrain transitive packages to the tested versions.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

On macOS/Linux, activate with `source .venv/bin/activate` instead. The supplied
`environment.yml` is an alternative conda setup using the same requirements.

## Train and evaluate

```console
python train_model.py
python -m pytest -v
python -m flake8 .
```

Training uses `data/census.csv`, a stratified 80/20 split with seed 42, and a
random forest. Encoders are fitted only on training rows and reused unchanged
for evaluation and serving. Whitespace is stripped, `?` remains an explicit
unknown category, and previously unseen categorical values are supported.

The pipeline writes `model/model.pkl`, `model/encoder.pkl`, `model/lb.pkl`,
`model/metrics.json`, and `slice_output.txt`. Slice metrics cover every observed
value of all eight categorical features in the test set. Precision, recall,
and F1 use `>50K` as the positive class; undefined metrics are reported as zero.
Only load trusted pickle/joblib artifacts. Keep the dependency versions used
to train the artifacts when serving them.

## Run the API

```console
python -m uvicorn main:app --reload
```

In a second terminal with the same environment:

```console
python local_api.py
```

The script prints status codes and results for a real GET and POST. Interactive
API documentation is at http://127.0.0.1:8000/docs.

| Route | Purpose |
| --- | --- |
| `GET /` | Welcome message |
| `POST /predict` | Income prediction for one Census record |
| `GET /health` | Process health |
| `GET /ready` | Model readiness |

Example request body:

```json
{
  "age": 39, "workclass": "State-gov", "fnlgt": 77516,
  "education": "Bachelors", "education-num": 13,
  "marital-status": "Never-married", "occupation": "Adm-clerical",
  "relationship": "Not-in-family", "race": "White", "sex": "Male",
  "capital-gain": 2174, "capital-loss": 0, "hours-per-week": 40,
  "native-country": "United-States"
}
```

The response contains `prediction`, either `<=50K` or `>50K`. Invalid or missing
features receive HTTP 422. Missing artifacts cause application startup to fail.
The default artifact directory is resolved relative to the project, and can be
overridden with the `MODEL_DIR` environment variable.

## Project files

- `ml/data.py` and `ml/model.py`: shared preprocessing and ML functions.
- `train_model.py`: reproducible training and slice evaluation.
- `main.py` and `app/`: API and validated input schema.
- `test_ml.py` and `tests/`: ML/data tests and API integration tests.
- `local_api.py`: live HTTP client.
- `model_card.md`: intended use, data, measured performance, and limitations.
- `.github/workflows/ci.yml`: pytest, flake8, training, and live API checks.
- `screenshots/`: required submission evidence.
- `docs/deployment.md`: local and container deployment instructions.

## Submission

The repository must be public. Include this repository URL in the submission
details. Check `docs/submission_checklist.md` before submitting. Required
screenshots are `unit_test.png`, `local_api.png`, and
`continuous_integration.png` inside `screenshots/`.

Dataset and starter attribution: [Udacity starter repository](https://github.com/udacity/Deploying-a-Scalable-ML-Pipeline-with-FastAPI).
This is an educational model, not a basis for consequential decisions about
individual people. Read the model card before interpreting predictions.
