import hashlib
import json
from pathlib import Path

from docx import Document
import nbformat
import numpy as np
import pandas as pd
import pymupdf


ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "outputs/modeling_summary.json").read_text())


def test_original_dataset_and_dimensions():
    raw = pd.read_csv(ROOT / "hour.csv")
    assert raw.shape == (17379, 17)
    assert hashlib.sha256((ROOT / "hour.csv").read_bytes()).hexdigest() == S["source_sha256"]
    assert raw.isna().sum().sum() == 0
    assert raw.duplicated(["dteday", "hr"]).sum() == 0
    assert (raw.casual + raw.registered).equals(raw.cnt)


def test_feature_exclusions_and_model_configuration():
    assert len(S["features"]) == 12
    assert set(S["features"]).isdisjoint({"cnt", "casual", "registered", "instant", "dteday"})
    assert S["forest_parameters"]["n_estimators"] == 150
    assert S["forest_parameters"]["max_depth"] == 14
    assert S["forest_parameters"]["random_state"] == 20261119
    assert all(v == 0 for v in S["test_unknown_category_counts"].values())


def test_chronological_complete_date_split():
    split = pd.read_csv(ROOT / "outputs/split_assignments.csv", parse_dates=["timestamp", "date"])
    assert len(split) == S["rows"]
    assert split.instant.is_unique
    assert split.groupby("date").partition.nunique().eq(1).all()
    dev = split.loc[split.partition.eq("development")]
    test = split.loc[split.partition.eq("test")]
    assert dev.timestamp.max() < test.timestamp.min()
    assert len(dev) == 13891 and len(test) == 3488
    assert str(test.date.min().date()) == "2012-08-07"


def test_cv_only_uses_earlier_development_with_day_gap():
    folds = json.loads((ROOT / "outputs/cv_folds.json").read_text())
    assert len(folds) == 3
    for fold in folds:
        assert pd.Timestamp(fold["validation_start"]) < pd.Timestamp("2012-08-07")
        assert (pd.Timestamp(fold["validation_start"]).normalize()
                - pd.Timestamp(fold["train_end"]).normalize()).days >= 2
    assert [f["train_rows"] for f in folds] == sorted(f["train_rows"] for f in folds)
    cv = pd.read_csv(ROOT / "outputs/development_cv.csv")
    assert cv.shape[0] == 6
    assert set(cv["model"]) == {"Training-mean baseline", "Random forest"}


def test_test_predictions_align_with_original_targets():
    raw = pd.read_csv(ROOT / "hour.csv").set_index("instant")
    pred = pd.read_csv(ROOT / "outputs/test_predictions.csv")
    np.testing.assert_array_equal(pred.actual, raw.loc[pred.instant, "cnt"])
    assert len(pred) == 3488 and pred.instant.is_unique
    assert np.isfinite(pred[["forest_prediction", "baseline_prediction"]]).all().all()
    assert pred.forest_prediction.ge(0).all()


def test_metrics_recomputed_without_sklearn():
    pred = pd.read_csv(ROOT / "outputs/test_predictions.csv")
    actual = pred.actual.to_numpy()
    for model, column in [("Random forest", "forest_prediction"),
                          ("Training-mean baseline", "baseline_prediction")]:
        row = next(r for r in S["metrics"]
                   if r["model"] == model and r["partition"] == "held_out_test")
        error = actual - pred[column].to_numpy()
        assert np.isclose(np.abs(error).mean(), row["mae"])
        assert np.isclose(np.sqrt(np.square(error).mean()), row["rmse"])
        assert np.isclose(1 - np.square(error).sum()
                          / np.square(actual - actual.mean()).sum(), row["r2"])


def test_baseline_uses_development_mean_not_test_mean():
    raw = pd.read_csv(ROOT / "hour.csv")
    mean = raw.loc[raw.dteday < "2012-08-07", "cnt"].mean()
    pred = pd.read_csv(ROOT / "outputs/test_predictions.csv")
    assert np.allclose(pred.baseline_prediction, mean)
    assert not np.isclose(mean, pred.actual.mean())


def test_hourly_errors_and_unrecorded_hours():
    pred = pd.read_csv(ROOT / "outputs/test_predictions.csv")
    hourly = pd.read_csv(ROOT / "outputs/test_error_by_hour.csv").set_index("hr")
    assert hourly.n.sum() == len(pred)
    assert set(hourly.index) == set(range(24))
    expected = (pred.actual - pred.forest_prediction).abs().groupby(pred.hr).mean()
    np.testing.assert_allclose(hourly.mae, expected)
    assert S["unrecorded_clock_hours"] == 165


def test_notebook_execution_and_embedded_figures():
    notebook = nbformat.read(ROOT / "modeling.ipynb", as_version=4)
    code = [c for c in notebook.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in code)
    assert not any(o.output_type == "error" for c in code for o in c.outputs)
    assert sum("image/png" in o.get("data", {}) for c in code for o in c.outputs) == 3
    assert "supervised learning" in notebook.cells[0].source


def test_final_summary_uses_actual_results():
    notebook = nbformat.read(ROOT / "modeling.ipynb", as_version=4)
    final = notebook.cells[-1]
    assert final.cell_type == "markdown" and "Notebook summary" in final.source
    assert len(final.source.split("\n\n")[1].split(".\n")) == 5
    assert "104.65" in final.source and "72.57" in final.source
    assert "SUMMARY_PENDING" not in "\n".join(c.source for c in notebook.cells)


def test_report_sections_figures_values_and_citations():
    doc = Document(ROOT / "module_summary.docx")
    text = "\n".join(p.text for p in doc.paragraphs)
    for heading in ["Overview", "Dataset Description", "Modeling Approach", "Results",
                    "Interpretation for a Non-Technical Audience",
                    "Limitations and Potential Bias", "References"]:
        assert heading in text
    for value in ["104.65", "72.57", "232.56", "0.775", "165", "55.00%"]:
        assert value in text
    assert len(doc.inline_shapes) == 3
    for name in ["Breiman", "Hodson", "Kapoor", "Fanaee-T"]:
        assert name in text
    assert "Instructor Name" not in text


def test_report_template_and_pdf():
    audit = json.loads((ROOT / "outputs/report_template_audit.json").read_text())
    assert audit["styles_preserved"] and audit["section_preserved"] and audit["headers_preserved"]
    with pymupdf.open(ROOT / "module_summary.pdf") as pdf:
        assert len(pdf) >= 5
        text = "\n".join(page.get_text() for page in pdf)
        assert "AI Foundations (Edward Hickman)" in text and "104.65" in text
        assert sum(len(page.get_images()) for page in pdf) >= 3
        assert "References" in text
        assert "10.1016/j.patter.2023.100804" in text


def test_required_files_and_frozen_packages():
    for name in ["modeling.ipynb", "module_summary.pdf", "requirements.txt", "hour.csv"]:
        assert (ROOT / name).is_file()
    requirements = (ROOT / "requirements.txt").read_text()
    for name in ["scikit-learn", "numpy", "pandas", "seaborn", "matplotlib"]:
        assert f"{name}==" in requirements
