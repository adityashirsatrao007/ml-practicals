# Advance Machine Learning Lab Manual

This document keeps the code, input, and output sections together so it can be exported to PDF later.

**Prepared by:** Aditya Shirsatrao  
**Mentor:** Prof. N. B. Aherwadi  
**Student Name:** Aditya Vishal Shirsatrao  
**Roll No:** 28

## Experiment 1: K-means Clustering

**Input:** UCI Wine dataset.

**Dataset Description:** 178 wine samples with 13 continuous physicochemical attributes and one class label. Class is not used during clustering.

**Code:** See [codes/exp1_kmeans.py](codes/exp1_kmeans.py)

**Output:** Scatter plot showing three clusters on the Iris data.

## Experiment 2: Hierarchical Clustering

**Input:** UCI Abalone dataset.

**Dataset Description:** 4,177 samples with shell measurements and weights. The categorical sex feature is encoded numerically for clustering.

**Code:** See [codes/exp2_hierarchical_clustering.py](codes/exp2_hierarchical_clustering.py)

**Output:** Dendrogram and cluster scatter plot.

## Experiment 3: Apriori Algorithm

**Input:** Groceries transaction dataset from a public GitHub mirror of open grocery basket data.

**Dataset Description:** Variable-length shopping transactions where each row lists purchased items; converted into one-hot format before Apriori mining.

**Code:** See [codes/exp3_apriori.py](codes/exp3_apriori.py)

**Output:** Frequent itemsets and association rules.

## Experiment 4: Market Basket Analysis

**Input:** UCI Adult dataset converted to market-basket transactions.

**Dataset Description:** Census records transformed into feature=value tokens (for example, education=Bachelors, income=>50K) to discover association rules.

**Code:** See [codes/exp4_market_basket_analysis.py](codes/exp4_market_basket_analysis.py)

**Output:** Frequent itemsets and rules for products bought together.

## Experiment 5: Reinforcement Learning

**Input:** Daily minimum temperatures dataset (time-series) used as reward signal for Q-learning transitions.

**Dataset Description:** Daily temperature changes are discretized into directional states, and rewards are assigned based on direction prediction correctness.

**Code:** See [codes/exp5_reinforcement_learning.py](codes/exp5_reinforcement_learning.py)

**Output:** Learned optimal policy.

## Experiment 6: Time Series Analysis

**Input:** Airline passengers time-series dataset from a public GitHub mirror of the open dataset.

**Dataset Description:** Monthly airline passenger counts from 1949-1960, suitable for stationarity checks, differencing, ARIMA fitting, and forecasting.

**Code:** See [codes/exp6_time_series_analysis.py](codes/exp6_time_series_analysis.py)

**Output:** Stationarity check, ACF/PACF plots, and ARIMA forecast.

## Experiment 7: Boosting

**Input:** UCI Breast Cancer Wisconsin (Diagnostic) dataset.

**Dataset Description:** 569 samples, 30 numeric features, binary diagnosis label (malignant/benign) for supervised classification.

**Code:** See [codes/exp7_boosting.py](codes/exp7_boosting.py)

**Output:** Cross-validation scores and AdaBoost accuracy.

## Notes

The repository now stores the standalone Python files in the [codes](/workspaces/Machine-Learning-Practicals/codes) folder. This Markdown file is the source used to generate the PDFs.

The generated PDF files are in [pdfs](/workspaces/Machine-Learning-Practicals/pdfs), one per experiment.