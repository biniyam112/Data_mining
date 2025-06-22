# Data Mining Project: EDA, Clustering, Prediction, and Sequential Pattern Mining

This project performs end-to-end data analysis on a customer-centered transaction dataset. It covers data understanding and preparation, unsupervised clustering for customer segmentation, supervised prediction of spending levels, and sequential pattern mining over purchase sequences.

## Learning Goals
- Understand data knowledge and discovery workflows.
- Assess and improve data quality.
- Engineer meaningful customer-centric features.
- Apply and evaluate clustering algorithms.
- Build and compare predictive models.
- Mine and interpret sequential patterns.
- Address ethical, privacy, and fairness considerations.

# Project Tasks

## Task 1: Data Understanding and Preparation
Explore, clean, and transform the data to build a reliable analytical base.

- Data semantics and schema
    - Describe entities, keys, and relationships.
    - Provide a concise data dictionary for key fields.
- Data quality assessment
    - Duplicates, missing values, inconsistent types, and outliers.
    - Handling strategies: imputation, deduplication, winsorization/capping, or removal with rationale.
- Exploratory data analysis (EDA)
    - Univariate distributions, summary statistics, skewness/kurtosis.
    - Bivariate analysis and pairwise correlations; identify multicollinearity.
    - Visualizations: histograms, KDEs, boxplots, heatmaps.
- Feature transformations and generation
    - Scaling/normalization; encoding of categorical variables.
    - Domain features for customer profile (examples):
        - RFM: Recency, Frequency, Monetary value.
        - Tenure, average basket size, basket diversity, discount propensity.
        - Visit interval stats, return rate, churn flags, seasonality indicators.
    - Log/Box-Cox transforms where appropriate.
- Redundancy reduction
    - Correlation filtering, variance threshold, VIF checks, or PCA for visualization.
- Train/test strategy
    - Define holdout or time-aware splits to prevent leakage.

Deliverables:
- Cleaned dataset, feature matrix, and a short EDA report with visuals and decisions.

## Task 2: Clustering Analysis (Customer Segmentation)
Segment customers based on their engineered profiles. Compare alternative algorithms and justify choices.

Preprocessing:
- Remove or combine highly correlated features.
- Standardize/normalize features consistently with Task 1.

Clustering approaches:
- K-means
    - Determine k (e.g., elbow, silhouette).
    - Interpret clusters using centroids and variable distributions vs. overall.
    - Evaluate with silhouette, Calinski–Harabasz, Davies–Bouldin; check stability.
- Density-based (e.g., DBSCAN/HDBSCAN)
    - Parameter tuning (eps/min_samples or min_cluster_size).
    - Handle noise/outliers; interpret cluster shapes.
    - Evaluate using internal metrics and percentage of noise points.
- Hierarchical clustering
    - Compare linkage methods (single, complete, average, Ward).
    - Choose optimal cut; discuss dendrograms and consistency across linkages.

Visualization and interpretation:
- Project clusters via PCA/UMAP for inspection.
- Provide business-oriented profiles per cluster.

Conclusions:
- Recommend the best approach and justify with metrics, stability, and interpretability.

Optional:
- Explore additional techniques from https://github.com/annoviko/pyclustering.

## Task 3: Predictive Analysis (Spending Tier Classification)
Predict whether each customer is a high-, medium-, or low-spending customer.

- Target definition
    - Compute labels from Monetary value or a composite score.
    - Use quantile thresholds or domain cutoffs; document methodology.
    - Ensure the target is nominal and avoid leakage.
- Modeling
    - Baselines: stratified majority and simple models.
    - Models to compare: Logistic Regression, Random Forest, Gradient Boosting, SVM, k-NN (or others).
    - Use pipelines for preprocessing (scaling/encoding) to keep splits clean.
    - Hyperparameter tuning with stratified cross-validation.
- Evaluation
    - Train and test performance with balanced metrics: macro-F1, accuracy, per-class recall; confusion matrices.
    - Handle class imbalance (class weights, resampling) if needed.
    - Calibration and feature importance/SHAP for interpretability.

Deliverables:
- Clear target definition, model comparison table, and discussion of trade-offs.

## Task 4: Sequential Pattern Mining
Mine frequent sequential patterns over customer purchasing behavior.

- Sequence modeling
    - Represent each customer as an ordered sequence of baskets/events.
    - Define sessionization rules (time gap or transaction grouping).
- Mining
    - Apply a sequential pattern mining algorithm (e.g., PrefixSpan, SPADE, GSP).
    - Choose minimum support (and optional confidence/lift for post-filtering).
- Interpretation
    - Summarize top patterns, their support, and business relevance.
    - Validate patterns against segments or time periods when applicable.

Optional:
- Extend with temporal constraints (e.g., max/min gap, time windows) or top-k mining.

# Ethics and Responsible AI
- Privacy: avoid re-identification; minimize sensitive attributes.
- Fairness: check differential performance across groups, if available.
- Transparency: document preprocessing, modeling choices, and limitations.

# What to Submit
- Reproducible notebooks/scripts for each task.
- Generated datasets/artifacts (cleaned features, labels, cluster assignments).
- Figures and tables supporting findings.
- A concise report summarizing methods, results, and recommendations.

# Environment (suggested)
- Python 3.x with common libraries: pandas, numpy, scikit-learn, scipy, matplotlib, seaborn, umap-learn, hdbscan (optional), and a sequential pattern mining library.
- Set random seeds; log versions for reproducibility.

# Notes
- Keep preprocessing consistent across tasks to avoid data leakage.
- Prefer simple, interpretable solutions when metrics are comparable.
- Document assumptions and decisions to ensure clarity and repeatability.
