# Machine Learning Foundations Capstone

**AI Foundations (Edward Hickman)**

This supervised regression project estimates recorded hourly Capital Bikeshare
rentals from calendar and observed weather inputs. A fixed random forest is
compared with a training-mean baseline using chronological validation and a
later held-out test period. It demonstrates conditional prediction, not causal
inference, demographic fairness or a deployed advance weather-based forecast.
Dataset: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset (CC BY 4.0).

## Reproduce

Use Python 3.11 (tested with 3.11.18), then from this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyterlab
```

Open `modeling.ipynb`, restart its kernel and run every cell in order.
Keep `hour.csv` beside the notebook; the analysis runs without network access
after dependencies are installed.

```bash
python -m nbconvert --to notebook --execute --inplace modeling.ipynb --ExecutePreprocessor.timeout=600
python -m pytest tests -q
pip freeze > requirements.txt
```

Required submission files: `modeling.ipynb`, `module_summary.pdf`, `hour.csv`,
`requirements.txt`. Editable report versions, saved outputs and tests are
included as supporting material. Read `SUBMISSION_CHECKLIST.md` and
`MODELING_PLAN.md` before submitting.
