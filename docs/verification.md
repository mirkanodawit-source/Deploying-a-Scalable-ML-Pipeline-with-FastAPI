# Verification record

Local verification was performed on October 14, 2026 with Python 3.11.16 and
the versions in `requirements-lock.txt`.

- Training completed on all 32,561 supplied rows with a 26,048/6,513 split.
- Held-out precision was 0.800525, recall 0.583546, and F1 0.675028.
- All 101 observed categorical slices were evaluated.
- All 24 tests passed, including ten ML/data tests and API integration tests.
- Flake8 passed without errors.
- The live GET and POST both returned HTTP 200; the example prediction was `<=50K`.
- The repository whitespace check passed.

Actual test and API output is preserved in `screenshots/unit_test.txt` and
`screenshots/local_api.txt`. One upstream Starlette/AnyIO deprecation warning
was emitted by TestClient; it did not affect test results.

The Dockerfile is supplied but was not built locally because Docker is not
installed. The required screenshot images are still pending after browser
capture was stopped. These outstanding items must not be confused with the
successful command-line checks above.
