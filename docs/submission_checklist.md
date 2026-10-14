# Submission checklist

Use actual run results and screenshots, not the examples from the assignment.

- [x] `python train_model.py` completes on the full provided Census dataset.
- [x] Saved model, encoder, and label binarizer are included in `model/`.
- [x] `slice_output.txt` includes all 101 observed categorical test slices.
- [x] Every section of `model_card.md` is complete with measured metrics.
- [x] `python -m pytest -v` passes: 24 tests, including ten ML/data tests.
- [x] `python -m flake8 .` passes.
- [x] `local_api.py` gets HTTP 200 for both GET and POST.
- [x] `screenshots/unit_test.png` shows ten passing ML tests in the console.
- [x] `screenshots/local_api.png` shows successful live GET and POST results.
- [x] `screenshots/continuous_integration.png` shows the passing GitHub workflow.
- [x] GitHub Actions passed lint, tests, training, and live API checks.
- [x] The implementation is pushed to the public fork linked in README.md.
- [ ] The same repository URL is entered in the submission details.
