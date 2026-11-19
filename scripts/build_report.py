"""Create the ML report from executed results using the supplied APA template."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

from docx import Document
from docx.enum.text import WD_BREAK
from docx.shared import Inches
from docx.text.paragraph import Paragraph
from lxml import etree


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
s = json.loads((OUT / "modeling_summary.json").read_text())
metrics = {(r["model"], r["partition"]): r for r in s["metrics"]}
f = metrics["Random forest", "held_out_test"]
b = metrics["Training-mean baseline", "held_out_test"]
train = metrics["Random forest", "development_training"]
dev, test = s["split"]
cv_forest = [r for r in s["cv_results"] if r["model"] == "Random forest"]
cv_rmse = ", ".join(f"{r['rmse']:.2f}" for r in cv_forest)

sections = [
    ("Overview",
     "This project uses supervised regression to estimate recorded hourly Capital "
     "Bikeshare rentals from calendar variables and contemporaneously observed weather. "
     "It trains a random-forest regressor on the real UCI Bike Sharing dataset "
     "and compares it with a training-mean baseline on later held-out dates "
     "(Fanaee-T, 2013; Fanaee-T & Gama, 2014). The dataset is available at "
     "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset; this evaluation "
     "does not demonstrate an advance forecast using weather forecasts."),
    ("Dataset Description",
     f"The unchanged hour.csv contains {s['rows']:,} recorded hours and "
     f"{s['raw_columns']} columns covering Washington, D.C.'s Capital Bikeshare "
     "system in 2011–2012, with weather and calendar information "
     "(Fanaee-T & Gama, 2014). The target cnt is total observed rental count; "
     "twelve predictors comprise season, year, month, hour, holiday, weekday, "
     "working-day status, weather category, temperature, apparent temperature, "
     "humidity and wind speed. The raw file has no missing cells or repeated "
     f"date/hour keys, but {s['unrecorded_clock_hours']} clock hours are absent "
     "from its complete time span and are not invented as zero observations. "
     "The raw columns also include a row identifier, date and the casual/registered "
     "rental subtotals, which are not model inputs. The source Readme and actual "
     "file agree on 17,379 records, while the UCI landing page's 17,389 count "
     "is inconsistent; all computations use the included file."),
    ("Modeling Approach",
     "The notebook loads the local CSV, displays its first rows and data types, "
     "and verifies its fingerprint, missingness, key uniqueness and relevant ranges. "
     "The twelve input features exclude the target, the row identifier and raw date. "
     "Crucially, casual and registered are excluded because their sum is exactly "
     "cnt; including them would reveal the answer rather than demonstrate useful "
     "prediction. This explicit feature-availability audit addresses leakage risks "
     "described by Kapoor and Narayanan (2023).\n\n"
     "A ColumnTransformer one-hot encodes eight categorical fields and passes "
     "through four numerical weather fields. Encoders are fitted on each training "
     "subset only; unseen categories use all-zero encodings rather than categories "
     "learned from validation or test data. No missing-value imputation is needed "
     "for the verified input. Extra scaling is not used for tree thresholds; the "
     "provider-normalized weather values remain unchanged. Those released "
     "normalizations are source preprocessing, not transformations newly fitted "
     "on the full dataset by this analysis.\n\n"
     "The last 20% of distinct dates, rounded up, form the test set; entire days "
     "remain together. Development covers "
     f"{dev['start'][:10]}–{dev['end'][:10]} ({dev['rows']:,} rows, {dev['dates']} dates), "
     f"and testing covers {test['start'][:10]}–{test['end'][:10]} "
     f"({test['rows']:,} rows, {test['dates']} dates). Three expanding-window "
     "development folds each leave one whole calendar day between training "
     "and validation. Fresh preprocessing and models are fitted within every fold. "
     "Chronology prevents future training relative to each evaluated period; it "
     "does not make temporally correlated records independent or make one future "
     "period representative of all deployment conditions (Kapoor & Narayanan, 2023).\n\n"
     "The fixed RandomForestRegressor averages 150 bootstrapped regression trees, "
     "uses squared-error splitting, a maximum depth of 14 and a minimum leaf "
     "size of two, with seed 20261119 and a single worker. All available transformed "
     "features are considered at each split (max_features=1.0). Tree ensembles "
     "can represent nonlinear relationships and interactions without specifying "
     "a linear functional form (Breiman, 2001). Depth and leaf constraints limit "
     "complexity but do not prove the model generalizes. The training-mean "
     "DummyRegressor is a transparent, deliberately simple comparison, not an "
     "optimized seasonal forecasting benchmark. No hyperparameter search or "
     "test-based model selection was performed; both models were refitted on "
     "all development records before final evaluation.\n\n"
     "RMSE is the primary metric because squared errors emphasize large rental-count "
     "misses; taking the root returns the error to rentals per recorded hour. "
     "MAE supplies a complementary mean absolute miss in the same units. These "
     "losses express different priorities; neither is universally superior "
     "(Hodson, 2022). R² is 1 minus the model's sum of squared errors divided "
     "by that of the evaluation period's actual mean; it can be negative and "
     "is not percent accuracy (scikit-learn developers, n.d.). The evaluation "
     "uses empirical losses, not a claim that errors are normally distributed, "
     "independent or homoscedastic."),
    ("Results",
     f"The forest's development-fold RMSE values were {cv_rmse} rentals per "
     "recorded hour. All three folds are retained in the notebook and saved "
     "table; their variation reflects different training sizes and time periods, "
     "not independent repetitions of the same experiment.\n\n"
     f"On the {f['n']:,} later held-out records, the forest achieved RMSE "
     f"{f['rmse']:.2f}, MAE {f['mae']:.2f} and R² {f['r2']:.3f}. The training-mean "
     f"baseline achieved RMSE {b['rmse']:.2f}, MAE {b['mae']:.2f} and R² "
     f"{b['r2']:.3f}. Figure 1 shows the resulting "
     f"{s['rmse_improvement_over_baseline']:.2%} RMSE reduction relative to this "
     "baseline. The negative baseline R² is possible because the development "
     "mean used for predictions differs from the test period's mean.\n\n"
     f"Forest training RMSE was {train['rmse']:.2f}, training MAE "
     f"{train['mae']:.2f} and training R² {train['r2']:.3f}, substantially better "
     "than the held-out scores. This gap is consistent with limited generalization "
     "and possibly overfitting, but time-period differences prevent attributing "
     f"it entirely to overfitting. Mean recorded rentals rose from "
     f"{s['development_mean_actual']:.2f} in development to "
     f"{s['test_mean_actual']:.2f} in testing.\n\n"
     "Figure 2 plots every held-out actual/predicted pair, rather than a selected "
     "successful week, and exposes departures from the exact-prediction diagonal. "
     f"The mean actual-minus-predicted residual was {s['test_mean_residual']:.2f}, "
     "so the model underpredicted the test period on average. Figure 3 shows "
     f"the largest clock-hour MAE at hour {s['worst_hour_by_mae']:02d}:00, "
     f"with {s['worst_hour_mae']:.2f} rentals per recorded hour. Group counts "
     "are labeled so readers can inspect coverage. No final-test categorical "
     "values were absent from the full development encoder's categories."),
    ("Interpretation for a Non-Technical Audience",
     f"The model's absolute miss averaged about {f['mae']:.0f} rentals per "
     "recorded hour in the later test period; individual errors could be much "
     "larger or smaller. It performed better than using one historical average "
     "for every hour, but this does not show that it beats a carefully designed "
     "seasonal baseline or every alternative model. Its test R² is not an "
     f"accuracy of {f['r2']:.1%}; it summarizes squared-error performance relative "
     "to a particular reference prediction. Large average errors at some hours "
     "and overall underprediction could matter for staffing or bike allocation, "
     "so a favorable overall score alone would not justify automated decisions. "
     "Because the test supplied observed weather, the scores should not be "
     "advertised as the performance of an advance forecasting service."),
    ("Limitations and Potential Bias",
     "The two historical years represent one city's system and recorded rentals, "
     "not all travel needs. Unmet demand, station outages and people unable to "
     "access the service are not directly measured, and the absence of a row "
     "does not establish zero demand. Using observed rentals to withdraw service "
     "from low-count areas could reinforce existing access differences; the file "
     "contains neither station-level nor demographic information sufficient to "
     "test geographic or demographic fairness. A proposed mitigation is to "
     "collect service-availability and station/neighborhood coverage data, audit "
     "allocation impacts and retain human review before deployment.\n\n"
     "Only one later test block is evaluated, and its seasonal and usage mix "
     "differs from development. Temporal dependence, changing membership, "
     "extreme weather or service changes may invalidate future performance "
     "expectations. The fixed random forest has limited extrapolation ability, "
     "and correlated calendar/weather inputs make simple feature-importance "
     "rankings difficult to interpret causally. No feature-importance, causal "
     "or prediction-interval claims are made. The one-day development gap is "
     "a precaution, not evidence that all dependence has vanished.\n\n"
     "Chronological separation, excluded outcome-derived columns and training-only "
     "encoding reduce identifiable leakage risks, following Kapoor and Narayanan "
     "(2023). They do not validate weather availability for advance use: a real "
     "forecast would require weather forecasts available at prediction time "
     "and a new prospective evaluation. Future work should also compare stronger "
     "seasonal baselines, evaluate later rolling periods and investigate large "
     "hour-specific errors. RMSE/MAE do not establish appropriate business loss "
     "weights or error distributions by themselves (Hodson, 2022)."),
    ("Reproducibility and Responsible Practice",
     "The package includes the original CSV and fingerprint, the executed "
     "modeling.ipynb, saved split assignments, fold boundaries, predictions, "
     "metrics and three figures. requirements.txt was generated by pip freeze, "
     "and the README explains dependency installation and restarting the kernel "
     "before running all cells. Fixed model parameters and random seeds make "
     "the computational choices explicit; package pins are not a guarantee "
     "of identical results on every operating system. Re-execution of the "
     "unchanged notebook is a reproducibility check, not additional test-driven "
     "tuning.\n\n"
     "Replit AI assisted with planning, coding, debugging, verification and report "
     "drafting. All reported metrics come from executed outputs; no hypothetical "
     "experiments are presented as results. The learner remains responsible "
     "for reviewing, understanding and revising the work and complying with "
     "the course's academic-integrity and AI-use policies."),
    ("Conclusion",
     "The fixed forest improved on the training-mean baseline under the documented "
     "later-period evaluation, but appreciable errors, underprediction and a "
     "training/test gap remain. The central lesson is to define the prediction "
     "setting and preserve feature and time boundaries before interpreting a "
     "favorable metric. This is a transparent educational ML workflow, not "
     "a validated causal explanation or production bike-allocation system."),
]

references = [
    ("Breiman, L. (2001). Random forests. Machine Learning, 45, 5–32. "
     "https://doi.org/10.1023/A:1010933404324", "Machine Learning, 45"),
    ("Fanaee-T, H. (2013). Bike Sharing [Data set]. UCI Machine Learning Repository. "
     "https://doi.org/10.24432/C5W894", "Bike Sharing"),
    ("Fanaee-T, H., & Gama, J. (2014). Event labeling combining ensemble detectors "
     "and background knowledge. Progress in Artificial Intelligence, 2, 113–127. "
     "https://doi.org/10.1007/s13748-013-0040-3", "Progress in Artificial Intelligence, 2"),
    ("Hodson, T. O. (2022). Root-mean-square error (RMSE) or mean absolute error "
     "(MAE): When to use them or not. Geoscientific Model Development, 15, "
     "5481–5487. https://doi.org/10.5194/gmd-15-5481-2022",
     "Geoscientific Model Development, 15"),
    ("Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis "
     "in machine-learning-based science. Patterns, 4(9), 100804. "
     "https://doi.org/10.1016/j.patter.2023.100804", "Patterns, 4"),
    ("scikit-learn developers. (n.d.). r2_score. "
     "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html",
     "r2_score"),
]

parser = argparse.ArgumentParser()
parser.add_argument("template", type=Path)
args = parser.parse_args()
doc = Document(args.template)
patterns = [deepcopy(p._p) for p in doc.paragraphs]
original_styles = doc.styles.element.xml
original_section = etree.tostring(doc.sections[0]._sectPr, method="c14n", exclusive=True)
header_footer = {str(p.partname): p.blob for p in doc.part.package.parts
                 if "header" in p.content_type or "footer" in p.content_type}
body = doc._element.body
for child in list(body):
    if child is not doc.sections[0]._sectPr:
        body.remove(child)


def append_pattern(index, text=""):
    """Replace sample content without changing a template paragraph's formatting."""
    element = deepcopy(patterns[index])
    original = Paragraph(element, doc._body)
    props = deepcopy(original.runs[0]._r.rPr) if original.runs else None
    for child in list(element):
        if child is not element.pPr:
            element.remove(child)
    body.insert(len(body) - 1, element)
    paragraph = Paragraph(element, doc._body)
    run = paragraph.add_run(text)
    if props is not None:
        run._r.insert(0, props)
    return paragraph


def page_break():
    """Insert a template-formatted page break."""
    append_pattern(13).runs[0].add_break(WD_BREAK.PAGE)


cover = {4: "Hourly Bike-Rental Prediction",
         6: "AI Foundations (Edward Hickman)", 7: "Udacity",
         8: "Project 3 — Machine Learning Foundations", 9: "", 10: ""}
for index in range(11):
    append_pattern(index, cover.get(index, ""))
page_break()
append_pattern(14, cover[4])
text = f"# {cover[4]}\n\n{cover[6]}\n\n"
for heading, content in sections:
    append_pattern(19, heading)
    for paragraph in content.split("\n\n"):
        append_pattern(15, paragraph)
    text += f"## {heading}\n\n{content}\n\n"

figures = [
    ("figure_1_test_rmse.png", "Both scores use the same held-out later records. "
     "Lower RMSE is better; the baseline predicts the development-period mean."),
    ("figure_2_actual_vs_predicted.png",
     "Every held-out record is plotted. The dashed line denotes exact predictions, "
     "not a fitted trend. Points below it represent underprediction."),
    ("figure_3_error_by_hour.png",
     "Bars show held-out mean absolute error; labels show the number of records "
     "in each clock-hour group. This is not a demographic fairness audit."),
]
for number, (filename, caption) in enumerate(figures, 1):
    page_break()
    append_pattern(19, f"Figure {number}")
    append_pattern(1).runs[0].add_picture(str(OUT / filename), width=Inches(6.5))
    append_pattern(15, f"Note. {caption} Source: executed modeling.ipynb.")

page_break()
append_pattern(37, "References")
text += "## References\n\n"
for reference, italic_span in references:
    paragraph = append_pattern(38)
    props = deepcopy(paragraph.runs[0]._r.rPr)
    for run in list(paragraph.runs):
        paragraph._p.remove(run._r)
    before, after = reference.split(italic_span, 1)
    for value, italic in [(before, False), (italic_span, True), (after, False)]:
        run = paragraph.add_run(value)
        if props is not None:
            run._r.insert(0, deepcopy(props))
        run.italic = italic
    text += f"{reference}\n\n"

doc.core_properties.title = cover[4]
doc.core_properties.author = cover[6]
doc.core_properties.subject = cover[8]
doc.core_properties.comments = "AI assistance disclosed. Instructor and due date were not supplied."
assert doc.styles.element.xml == original_styles
assert etree.tostring(doc.sections[0]._sectPr, method="c14n",
                      exclusive=True) == original_section
assert header_footer == {str(p.partname): p.blob for p in doc.part.package.parts
                         if "header" in p.content_type or "footer" in p.content_type}
doc.save(ROOT / "module_summary.docx")
(ROOT / "module_summary.md").write_text(text)
subprocess.run([
    "libreoffice", "-env:UserInstallation=file:///tmp/ml-report-office",
    "--headless", "--convert-to", "pdf", "--outdir", str(ROOT),
    str(ROOT / "module_summary.docx")], check=True)
(OUT / "report_template_audit.json").write_text(json.dumps({
    "template_url": "https://docs.google.com/document/d/1cv_MjQVKnGc9sbaYiOUna2155jMEvUxACwyDax-XYWI/edit",
    "template_sha256": hashlib.sha256(args.template.read_bytes()).hexdigest(),
    "styles_preserved": True, "section_preserved": True, "headers_preserved": True,
    "omitted_fields": ["Instructor", "Due date"],
    "pdf_font": "Liberation Sans substitutes for retained Arial DOCX styles",
}, indent=2) + "\n")
