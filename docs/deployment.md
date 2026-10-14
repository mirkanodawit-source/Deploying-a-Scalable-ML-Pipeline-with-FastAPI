# Deployment

Use Python 3.11 and install `requirements.txt` for serving or
`requirements-dev.txt` for development. Run all commands from the repository root.

1. Run `python train_model.py` to create the model, feature encoder, and label
   binarizer together. Keep these files as one bundle.
2. Run `python -m uvicorn main:app --host 127.0.0.1 --port 8000`.
3. Run `python local_api.py` in a second terminal.
4. Check `/ready` for model availability and `/docs` for the input schema.

The app loads artifacts once at startup. It fails startup if artifacts are
missing or the bundle has inconsistent feature counts/classes. Restart the
server after replacing a model bundle. `MODEL_DIR` optionally overrides the
default project-relative `model/` directory. The application does not
automatically read `.env` files.

## Container

```console
docker build -t census-api .
docker run --rm -p 8000:8000 census-api
```

The image serves on port 8000 as a non-root user. The model bundle is copied
into the image. Rebuild after retraining. The rubric requires local API execution;
a public production deployment would additionally need suitable access controls,
TLS, request limits, and monitoring.

## Continuous integration

Pushes to `main` or `master`, pull requests, and manual dispatch run CI. The job
installs pinned dependencies, lints, tests the committed bundle, runs a fresh
training pipeline in a temporary directory, and performs live GET/POST requests.
CI uses read-only repository permissions.
