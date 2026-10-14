# Model Card

## Model Details

This educational Census income classifier was trained on October 14, 2026.
It uses scikit-learn 1.6.1's RandomForestClassifier with 150 trees, maximum
depth 14, minimum leaf size 2, seed 42, and one worker. The model receives six
numeric features and one-hot encodings of eight categorical features. The
positive class is income `>50K`; the negative class is `<=50K`.

The implementation and artifacts are in
[the project fork](https://github.com/mirkanodawit-source/Deploying-a-Scalable-ML-Pipeline-with-FastAPI).
The training command is `python train_model.py`. The feature encoder, label
binarizer, and classifier are saved together in `model/`. This is an untuned
baseline; no hyperparameters were selected using the held-out test metrics.

## Intended Use

The model demonstrates reproducible preprocessing, training, slice evaluation,
unit tests, and local FastAPI inference for an ML engineering course. Intended
users are students and reviewers working with Census-like tabular examples.
It is not validated for present-day income estimation, lending, hiring,
insurance, eligibility, or other consequential decisions about individuals.

## Training Data

The project uses the 32,561-row `data/census.csv` supplied by the
[Udacity starter repository](https://github.com/udacity/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/tree/main/data).
It is the Adult/Census income dataset, with fourteen predictors and a `salary`
label. The file SHA-256, with line endings normalized to LF, is
`65d646e4156341692416e24dd1015146425d9d2cbdea2a3cc1fc3025da804e44`.

A stratified 80/20 random split with seed 42 provides 26,048 training rows.
Column names and string values have surrounding whitespace stripped. Missing
categorical markers `?` are retained as an explicit category; numeric features
must be finite. Rows, including any duplicates, are retained as supplied.
The one-hot encoder and label binarizer are fitted only on the training split.
The six numeric features are age, fnlgt, education-num, capital-gain,
capital-loss, and hours-per-week. The eight categorical features are workclass,
education, marital-status, occupation, relationship, race, sex, and native-country.

## Evaluation Data

The held-out split contains 6,513 rows, including 1,568 positive-income rows.
It uses the training encoders without refitting. The split is random and
stratified rather than temporal, and is drawn from the same source as training.
It does not measure performance on a new population or contemporary data.
No separate external validation dataset was used. Retaining duplicate rows
means a random row split may place identical records on both sides; these
results should therefore be treated as within-dataset baseline estimates.

## Metrics

Precision, recall, and F1 measure the `>50K` class with the classifier's default
decision rule. Undefined metrics are set to zero, including slices with no
positive examples or no positive predictions.

| Metric | Held-out result |
| --- | ---: |
| Precision | 0.8005 |
| Recall | 0.5835 |
| F1 | 0.6750 |

The pipeline evaluates 101 observed values across all eight categorical
features. Full results and slice counts are in `slice_output.txt` and
`model/metrics.json`. Selected slices illustrate variation:

| Feature value | Rows | Positive rows | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| sex: Female | 2,158 | 245 | 0.8759 | 0.5184 | 0.6513 |
| sex: Male | 4,355 | 1,323 | 0.7896 | 0.5956 | 0.6790 |
| race: Black | 662 | 93 | 0.8214 | 0.4946 | 0.6174 |
| race: White | 5,533 | 1,410 | 0.8050 | 0.5887 | 0.6800 |

Recall of 0.5835 means the model misses approximately 42% of positive test
examples. Slice results are descriptive, not a fairness certification; some
slices have very few positive examples and unstable metrics.

## Ethical Considerations

The dataset reflects historical income inequalities and representation limits.
Race and sex are included, alongside features that can act as proxies for
protected attributes. Predictions can reproduce patterns of discrimination
present in the data. Dataset categories are limited and do not describe every
person's identity. Differences between slice metrics require investigation,
and good aggregate metrics do not establish equitable treatment. No personal
data should be submitted to a public demo without an appropriate privacy basis.

## Caveats and Recommendations

The model is limited to the supplied historical population and label definition.
Income thresholds and feature distributions can shift over time. Unseen
categorical values are encoded as zero indicator groups, which permits
inference but does not establish reliability for those groups. Small slices,
class imbalance, and the lack of confidence intervals limit interpretation.

Before any use beyond coursework, obtain representative external validation
data, assess subgroup and intersectional performance, quantify uncertainty,
review feature suitability and privacy, and monitor drift. The API validates
types and ranges but does not verify that all feature combinations are
semantically consistent. Only load trusted serialized model artifacts and
keep the recorded dependency versions consistent between training and serving.
