from __future__ import annotations

import html
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = ROOT / "pdf_build"
PDF_DIR = ROOT / "pdfs"


EXPERIMENTS = [
    {
        "number": 1,
        "title": "Implementing K-means Clustering",
        "aim": "To study the implementation of K-means clustering algorithm.",
    "input": "UCI Wine dataset (13 chemical features and class label).",
    "dataset_description": "Contains 178 wine samples with 13 continuous physicochemical attributes. The class label is ignored for unsupervised clustering.",
    "output": "Scatter plot of 3 clusters after PCA projection to 2D.",
    "dataset_source": "UCI Machine Learning Repository - Wine",
        "code": ROOT / "codes" / "exp1_kmeans.py",
        "figure": ROOT / "assets" / "kmeans-clustering.svg",
    "extra_figure": ROOT / "assets" / "kmeans-elbow.svg",
    "extra_caption": "Elbow curve used to justify cluster count.",
    },
    {
        "number": 2,
        "title": "Implementing Hierarchical Clustering",
        "aim": "To study agglomerative hierarchical clustering and dendrogram plotting.",
    "input": "UCI Abalone dataset with measurements and one encoded categorical field.",
    "dataset_description": "Contains 4,177 abalone records. Features include shell measurements and weights; the sex category is encoded numerically for clustering.",
        "output": "Dendrogram and cluster scatter plot.",
    "dataset_source": "UCI Machine Learning Repository - Abalone",
        "code": ROOT / "codes" / "exp2_hierarchical_clustering.py",
        "figure": ROOT / "assets" / "hierarchical-clustering.svg",
      "extra_figure": ROOT / "assets" / "hierarchical-linkage-distances.svg",
      "extra_caption": "Cluster merge distance profile from hierarchical linkage.",
    },
    {
        "number": 3,
        "title": "Implementation of Apriori Algorithm",
        "aim": "To study frequent itemset mining using the Apriori algorithm.",
    "input": "Groceries transaction dataset from a public open-source mirror.",
    "dataset_description": "Each row is a basket transaction with variable-length item lists. Data is one-hot encoded using TransactionEncoder before mining frequent itemsets.",
        "output": "Frequent itemsets and association rules.",
    "dataset_source": "Public GitHub mirror of groceries data",
        "code": ROOT / "codes" / "exp3_apriori.py",
        "figure": ROOT / "assets" / "apriori-itemsets.svg",
      "extra_figure": ROOT / "assets" / "apriori-support-distribution.svg",
      "extra_caption": "Support distribution of top frequent itemsets.",
    },
    {
        "number": 4,
        "title": "Implementation of Market Basket Analysis",
        "aim": "To study association rule mining for shopping basket data.",
    "input": "UCI Adult dataset converted into market-basket style transactions.",
    "dataset_description": "Adult census rows are transformed into transactions like feature=value tokens (workclass, education, occupation, income, etc.) for association rule analysis.",
        "output": "Frequent itemsets and rules for products bought together.",
    "dataset_source": "UCI Machine Learning Repository - Adult",
        "code": ROOT / "codes" / "exp4_market_basket_analysis.py",
        "figure": ROOT / "assets" / "market-basket-analysis.svg",
      "extra_figure": ROOT / "assets" / "market-basket-lift.svg",
      "extra_caption": "Lift values of top association rules.",
    },
    {
        "number": 5,
        "title": "Reinforcement Learning",
        "aim": "To study reward calculation, discounted reward, optimal action selection, and Q-learning.",
    "input": "Daily minimum temperatures time-series used to derive state transitions and rewards.",
    "dataset_description": "Temperature deltas are bucketed into discrete states (decrease, stable, increase). Reward is assigned by matching predicted vs actual direction change.",
    "output": "Discounted reward value and learned optimal policy over state buckets.",
    "dataset_source": "Public GitHub mirror - Daily Minimum Temperatures",
        "code": ROOT / "codes" / "exp5_reinforcement_learning.py",
        "figure": ROOT / "assets" / "reinforcement-learning.svg",
      "extra_figure": ROOT / "assets" / "reinforcement-reward-trend.svg",
      "extra_caption": "Episode-wise reward trend during policy improvement.",
    },
    {
        "number": 6,
        "title": "Time Series Analysis",
        "aim": "To study stationarity, differencing, ADF test, ACF/PACF, ARIMA, and forecasting.",
    "input": "Airline passengers time-series dataset from a public open-source mirror.",
    "dataset_description": "Monthly passenger counts from 1949 to 1960. Series is non-stationary, making it suitable for differencing, ADF testing, and ARIMA forecasting.",
        "output": "Stationarity check, ACF/PACF plots, and ARIMA forecast.",
        "dataset_source": "Public GitHub mirror of the Airline Passengers dataset",
        "code": ROOT / "codes" / "exp6_time_series_analysis.py",
        "figure": ROOT / "assets" / "time-series-forecast.svg",
        "extra_figure": ROOT / "assets" / "time-series-acf-pacf.svg",
        "extra_caption": "ACF and PACF diagnostic patterns for ARIMA selection.",
    },
    {
        "number": 7,
        "title": "Boosting",
        "aim": "To study cross-validation and AdaBoost classification.",
    "input": "UCI Breast Cancer Wisconsin (Diagnostic) dataset.",
    "dataset_description": "Contains 569 samples with 30 real-valued features extracted from digitized images of breast mass. Target is malignant vs benign diagnosis.",
        "output": "Cross-validation scores and AdaBoost accuracy.",
    "dataset_source": "UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic)",
        "code": ROOT / "codes" / "exp7_boosting.py",
        "figure": ROOT / "assets" / "boosting-results.svg",
      "extra_figure": ROOT / "assets" / "boosting-confusion-matrix.svg",
      "extra_caption": "Confusion matrix after AdaBoost classification.",
    },
]

STUDENT_NAME = "Aditya Vishal Shirsatrao"
ROLL_NO = "28"
PREPARED_BY = "Aditya Shirsatrao"
MENTOR = "Prof. N. B. Aherwadi"


def escape_code(text: str) -> str:
    return html.escape(text)


def build_html(exp: dict) -> str:
    code = escape_code(exp["code"].read_text())
    figure_uri = exp["figure"].resolve().as_uri()
    figures_html = f"""
      <figure>
        <img src=\"{figure_uri}\" alt=\"Experiment {exp['number']} primary plot\" />
        <figcaption>Primary output plot</figcaption>
      </figure>
    """
    if exp.get("extra_figure"):
        extra_uri = exp["extra_figure"].resolve().as_uri()
        extra_caption = html.escape(exp.get("extra_caption", "Additional diagnostic plot"))
        figures_html += f"""
      <figure>
        <img src=\"{extra_uri}\" alt=\"Experiment {exp['number']} additional plot\" />
        <figcaption>{extra_caption}</figcaption>
      </figure>
    """
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Experiment {exp['number']}: {html.escape(exp['title'])}</title>
  <style>
    @page {{ margin: 14mm 12mm; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #fff; color: #000; font-family: Georgia, 'Times New Roman', serif; }}
    .sheet {{ width: 100%; max-width: 850px; margin: 0 auto; background: #fff; padding: 26px 30px 34px; border: 1px solid #000; box-shadow: none; }}
    .masthead {{ text-align: center; border-bottom: 2px solid #000; padding-bottom: 14px; margin-bottom: 18px; }}
    .masthead .dept {{ font-size: 14px; letter-spacing: .5px; text-transform: uppercase; font-family: Arial, Helvetica, sans-serif; color: #000; }}
    .masthead h1 {{ margin: 8px 0 2px; font-size: 24px; color: #000; }}
    .masthead h2 {{ margin: 0; font-size: 19px; font-weight: 600; color: #000; }}
    .meta {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px 18px; font-family: Arial, Helvetica, sans-serif; font-size: 12.5px; margin: 14px 0 18px; }}
    .meta div {{ padding: 8px 10px; background: #fff; border: 1px solid #000; border-radius: 4px; }}
    .student {{ display: flex; flex-wrap: wrap; gap: 10px 18px; font-family: Arial, Helvetica, sans-serif; font-size: 12.5px; margin: 0 0 18px; }}
    .student div {{ padding: 8px 10px; background: #fff; border: 1px solid #000; border-radius: 4px; }}
    .section {{ margin: 14px 0 18px; }}
    .section h3 {{ margin: 0 0 8px; font-size: 16px; color: #000; border-left: 4px solid #000; padding-left: 8px; }}
    p {{ margin: 0 0 8px; font-size: 13.5px; line-height: 1.5; }}
    .note {{ font-family: Arial, Helvetica, sans-serif; font-size: 12px; color: #000; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; background: #fff; color: #000; border: 1px solid #000; border-radius: 4px; padding: 14px 16px; font-size: 11px; line-height: 1.45; }}
    img {{ display: block; width: 100%; max-width: 100%; margin: 0 auto; border: 1px solid #000; border-radius: 4px; background: #fff; filter: grayscale(100%); }}
    figure {{ margin: 0 0 12px; }}
    figcaption {{ text-align: center; font-family: Arial, Helvetica, sans-serif; font-size: 11px; margin-top: 4px; }}
    ul {{ margin: 6px 0 0 18px; padding: 0; font-size: 13px; line-height: 1.45; }}
    li {{ margin: 0 0 4px; }}
    .footer {{ margin-top: 18px; padding-top: 10px; border-top: 1px solid #000; font-family: Arial, Helvetica, sans-serif; font-size: 11px; color: #000; text-align: center; }}
  </style>
</head>
<body>
  <div class="sheet">
    <div class="masthead">
      <div class="dept">Department of Artificial Intelligence &amp; Data Science Engineering</div>
      <h1>LABORATORY MANUAL</h1>
      <h2>Advance Machine Learning Lab</h2>
    </div>

    <div class="meta">
      <div><strong>Semester:</strong> VI</div>
      <div><strong>Subject Code:</strong> BTAIL606</div>
      <div><strong>Experiment:</strong> {exp['number']}</div>
      <div><strong>Title:</strong> {html.escape(exp['title'])}</div>
      <div><strong>Prepared by:</strong> {html.escape(PREPARED_BY)}</div>
      <div><strong>Mentor:</strong> {html.escape(MENTOR)}</div>
      <div><strong>Dataset source:</strong> {html.escape(exp['dataset_source'])}</div>
    </div>

    <div class="student">
      <div><strong>Student Name:</strong> {html.escape(STUDENT_NAME)}</div>
      <div><strong>Roll No:</strong> {ROLL_NO}</div>
    </div>

    <div class="section">
      <h3>Aim</h3>
      <p>{html.escape(exp['aim'])}</p>
    </div>

    <div class="section">
      <h3>Input</h3>
      <p>{html.escape(exp['input'])}</p>
    </div>

    <div class="section">
      <h3>Dataset Description</h3>
      <p>{html.escape(exp['dataset_description'])}</p>
    </div>

    <div class="section">
      <h3>Code</h3>
      <pre>{code}</pre>
    </div>

    <div class="section">
      <h3>Output</h3>
      <p>{html.escape(exp['output'])}</p>
      {figures_html}
    </div>

    <div class="section">
      <h3>Conclusion</h3>
      <p>Experiment {exp['number']} demonstrates the expected behavior of the algorithm using an open dataset or open source-compatible input.</p>
    </div>

    <div class="footer"></div>
  </div>
</body>
</html>
"""


def main() -> None:
    BUILD_DIR.mkdir(exist_ok=True)
    PDF_DIR.mkdir(exist_ok=True)

    for exp in EXPERIMENTS:
        html_path = BUILD_DIR / f"exp{exp['number']}.html"
        pdf_path = PDF_DIR / f"exp{exp['number']}_{exp['title'].lower().replace(' ', '_').replace('-', '_')}.pdf"
        html_path.write_text(build_html(exp), encoding="utf-8")
        subprocess.run(
            [
                "wkhtmltopdf",
                "--enable-local-file-access",
                "--page-size",
                "A4",
                "--margin-top",
                "10mm",
                "--margin-right",
                "10mm",
                "--margin-bottom",
                "10mm",
                "--margin-left",
                "10mm",
                str(html_path),
                str(pdf_path),
            ],
            check=True,
        )
        print(f"Created {pdf_path}")


if __name__ == "__main__":
    main()