# Required evidence

These images capture actual successful runs:

- `unit_test.png`: browser execution console showing output of
  `python -m pytest test_ml.py -v --color=no` with ten passing ML tests.
- `local_api.png`: browser execution console showing live GET and POST output
  from `python local_api.py` against the running API.
- `continuous_integration.png`: GitHub Actions page showing the passing workflow.

Do not substitute generated images or example outputs for execution evidence.

The local console launched the actual commands as subprocesses and displayed
their captured output without changing it. The screenshots were captured from
that display and converted from JPEG to PNG without altering their content.
The accompanying `unit_test.txt` and `local_api.txt` preserve command output.
