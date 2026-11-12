
# Customer Segmentation & Churn Risk

## Advanced Customer Propensity Modeling, Predictive Machine Learning and K-Means Segmentation for Targeted Retention Strategies

An integrated customer analytics project combining supervised Machine Learning, statistical feature analysis, hyperparameter optimization and unsupervised clustering to identify churn-related offer variables, predict customer attrition and segment a telecommunications customer base according to different levels of churn risk.

## Project Overview

### Objective: From Customer Churn Prediction to Targeted Retention Strategy

Customer churn represents a critical challenge for telecommunications companies, affecting recurring revenues, customer lifetime value and the sustainability of customer acquisition investments.

The project develops a quantitative framework to investigate the relationship between customer characteristics, subscribed services, contractual conditions and the probability of service cancellation.

The objective extends beyond predicting whether an individual customer will leave.

The methodology combines **predictive churn modeling with customer segmentation** to identify the variables most strongly associated with customer attrition and classify customers into groups with different risk profiles.

The framework addresses four analytical objectives:

- **Churn driver identification:** investigating the contractual, economic, demographic and service-related characteristics associated with customer attrition.
- **Predictive modeling:** comparing and optimizing classification algorithms to estimate churn propensity.
- **Customer segmentation:** applying Principal Component Analysis (PCA) and K-Means clustering to identify homogeneous customer groups.
- **Retention strategy:** translating predictive and segmentation results into differentiated customer retention and offer reinforcement priorities.

The distinctive contribution is the integration of **individual churn propensity analysis and statistically validated customer clustering**, connecting predictive analytics with actionable customer relationship management strategies.

## K-Means: Discovering Hidden Customer Segments for Precision Marketing

A distinctive component of this project is the use of **K-Means unsupervised learning to discover hidden customer segments beyond traditional, predefined marketing categories**.

Unlike conventional segmentation based on individual attributes such as age, expenditure or contract type, K-Means analyzes customers across multiple dimensions simultaneously, identifying patterns of similarity that emerge from the combined relationships among customer characteristics, subscribed services, spending behavior and contractual conditions.

This cross-dimensional approach allows the model to uncover **latent customer profiles that may not be immediately visible through standard descriptive analysis**, providing a data-driven foundation for identifying meaningful marketing targets.

In this project, K-Means is applied after PCA dimensionality reduction to identify **three distinct customer clusters**, which are subsequently analyzed against observed churn outcomes. The resulting segments exhibit substantially different churn rates: **8.06%, 43.94% and 15.18%**.

The strategic value lies in translating these hidden patterns into **actionable marketing segmentation**: identifying high-risk customer groups, differentiating retention priorities, investigating relevant offer characteristics and directing marketing investments toward more precisely defined targets.

By combining unsupervised pattern discovery with supervised churn prediction, the framework moves beyond identifying customers likely to leave and provides a methodology for understanding **which customer segments should be targeted, how they differ and where differentiated retention strategies may create value**.

## Analytical Framework

The methodology integrates three complementary components.

**Customer Propensity and Offer Analysis**

Multivariate statistical analysis investigates the relationships between churn, customer tenure, spending patterns, contract characteristics and subscribed services.

**Predictive Machine Learning**

Logistic Regression, Random Forest and XGBoost are evaluated and optimized to predict customer churn and identify influential explanatory variables.

**Customer Segmentation and Risk Classification**

PCA and K-Means clustering identify three customer segments. Their characteristics and churn distributions are subsequently analyzed and statistically tested to support differentiated retention strategies.

## Methodology

### 1. Customer Dataset and Churn Composition

The project uses the **Telco Customer Churn** dataset, attributed in the notebook to IBM Sample Data Sets.

The dataset contains **7,043 customer observations** and 21 original variables describing telecommunications subscriptions, customer characteristics and service cancellation.

The target variable is:

**Churn — Customer Service Cancellation (Yes / No)**

The explanatory variables cover several categories.

**Customer Relationship and Economic Variables**
- tenure — Duration of the customer relationship.
- MonthlyCharges — Monthly subscription expenditure.
- TotalCharges — Total accumulated customer expenditure.

**Contractual and Payment Characteristics**
- Contract — Month-to-month, one-year or two-year contracts.
- PaymentMethod — Payment method.
- PaperlessBilling — Electronic billing preference.

**Subscribed Services**
- PhoneService and MultipleLines.
- InternetService — DSL, fiber optic or no internet service.
- OnlineSecurity.
- OnlineBackup.
- DeviceProtection.
- TechSupport.
- StreamingTV and StreamingMovies.

**Customer Characteristics**
- gender.
- SeniorCitizen.
- Partner.
- Dependents.

### Customer Churn Distribution

| Customer Status | Customers | Percentage |
|---|---:|---:|
| Active customers | 5,174 | 73.46% |
| Churned customers | 1,869 | 26.54% |
| **Total** | **7,043** | **100%** |

Approximately **one in four customers has churned**.

The imbalance between active and churned customers is considered when defining the predictive modeling framework, including stratified train-test splitting.

This initial distribution provides the baseline against which the churn risk of individual customer segments can be evaluated.

### 2. Statistical Analysis of Customer Churn Drivers

The project investigates the relationships between customer attrition and explanatory variables using Pearson correlation, multivariate analysis and feature importance assessment.

Particular attention is given to customer tenure, expenditure, contractual conditions and the combination of subscribed services.

**Selected Pearson correlations with churn:**

| Variable | Correlation with Churn |
|---|---:|
| Month-to-month contract | +0.405 |
| Customer tenure | -0.352 |
| No Online Security | +0.343 |
| No Technical Support | +0.337 |
| Fiber Optic Internet | +0.308 |
| Two-year contract | -0.302 |
| Electronic Check Payment | +0.302 |
| No Online Backup | +0.268 |
| No Device Protection | +0.252 |

The analysis identifies several important relationships.

**Contractual Stability**

Month-to-month contracts show the strongest positive correlation with churn among the listed variables.

Conversely, longer contractual commitments are associated with lower churn.

**Customer Relationship Duration**

Tenure exhibits a negative correlation with churn, indicating that customers with longer relationships tend to have lower attrition rates.

**Digital Protection and Technical Services**

The absence of Online Security, Technical Support, Online Backup and Device Protection is positively associated with churn.

These results suggest that the adoption of protective and support services may be relevant to customer retention.

**Internet Service Characteristics**

Fiber optic subscriptions exhibit a positive association with churn in the analyzed customer base.

These relationships are statistical associations rather than evidence of causal effects.

### 3. Multicollinearity Assessment and Feature Selection

The project investigates correlations among explanatory variables to identify redundancy and improve the predictive feature set.

Strong relationships are observed between:

- Customer tenure and TotalCharges.
- MonthlyCharges and fiber optic subscriptions.
- Related service indicators generated through categorical encoding.

The analysis recognizes that one-hot encoded variables can introduce substantial redundancy, particularly when several binary indicators describe mutually exclusive categories.

A feature selection procedure evaluates categorical dummy groups and retains variables according to their relationship with the churn target.

The resulting modeling dataset contains **30 explanatory variables**, integrating customer characteristics, contractual conditions, expenditure and service subscriptions.

The objective is to retain informative customer attributes while reducing unnecessary overlap among explanatory variables.

### 4. Comparative Machine Learning Modeling

The project evaluates three supervised classification approaches.

**Logistic Regression**

Provides an interpretable statistical baseline for investigating the relationship between explanatory variables and churn probability.

**Random Forest Classifier**

Captures nonlinear relationships and interactions among customer characteristics, subscribed services and economic variables.

**XGBoost Classifier**

Uses gradient-boosted decision trees to investigate whether sequential ensemble learning improves classification performance.

The models are evaluated using:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion matrices.

Particular attention is given to the distinction between overall classification accuracy and the ability to identify customers who actually churn.

### Initial Model Performance

| Model | Test Accuracy |
|---|---:|
| Logistic Regression | 79.87% |
| Random Forest | 78.07% |
| XGBoost | 79.66% |

The comparison provides a foundation for subsequent model optimization.

### 5. Hyperparameter Optimization with Optuna

The project applies **Optuna** to optimize Random Forest and XGBoost classification models.

The optimization process evaluates alternative hyperparameter combinations using cross-validation.

**Random Forest Optimization**

The search explores:

- Number of decision trees.
- Maximum tree depth.
- Minimum samples required for splitting.
- Minimum samples per leaf.
- Maximum features considered at each split.

The notebook executes 10 Optuna trials using five-fold cross-validation and accuracy as the optimization criterion.

**Selected Random Forest configuration:**

| Hyperparameter | Value |
|---|---:|
| Number of estimators | 229 |
| Maximum depth | 8 |
| Minimum samples split | 2 |
| Minimum samples leaf | 5 |
| Maximum features | sqrt |

**Optimized Random Forest performance:**

| Metric | Result |
|---|---:|
| Accuracy | **80.3%** |
| Weighted Precision | 79.3% |
| Weighted Recall | 80.3% |
| Weighted F1-score | 79.5% |

**XGBoost Optimization**

The XGBoost model is also optimized with Optuna, using 10 trials and three-fold cross-validation.

The search includes:

- Number of estimators.
- Learning rate.
- Maximum tree depth.
- Subsample ratio.
- Column sampling ratio.

The optimized XGBoost model achieves:

| Metric | Result |
|---|---:|
| Accuracy | **80.94%** |
| Churn-class Precision | 68.28% |
| Churn-class Recall | 52.94% |
| Churn-class F1-score | 59.64% |

The optimized XGBoost achieves the highest reported overall accuracy.

The optimized Random Forest is retained in the notebook for subsequent customer profiling and segmentation analysis.

The distinction between overall accuracy and churn-class performance is important: predicting the majority of active customers correctly does not necessarily imply equally strong identification of customers at risk of leaving.

### 6. Feature Importance and Customer Churn Profile

The project analyzes the relative contribution of explanatory variables to the Random Forest churn classification.

**Selected Random Forest feature importance results:**

| Variable | Feature Importance |
|---|---:|
| TotalCharges | 0.1681 |
| tenure | 0.1459 |
| MonthlyCharges | 0.1438 |
| Month-to-month contract | 0.0663 |
| No Technical Support | 0.0314 |
| Electronic Check Payment | 0.0281 |
| Fiber Optic Internet | 0.0273 |
| No Online Security | 0.0253 |

The three most influential variables are:

**TotalCharges, tenure and MonthlyCharges.**

Together, they describe the economic value and duration of the customer relationship.

The results support a customer risk analysis that combines expenditure patterns, relationship maturity and service characteristics.

### Profile of Customers Predicted to Churn

The notebook derives an average profile for customers classified as churners.

The estimated values for the continuous variables are:

| Variable | Average Value |
|---|---:|
| Customer tenure | 9.56 months |
| MonthlyCharges | 77.42 |
| TotalCharges | 838.56 |

The profile describes customers with relatively short relationships and meaningful monthly expenditure.

The analysis also examines the frequency of categorical characteristics among customers predicted to churn.

**Selected characteristics of the predicted-churn group:**

| Characteristic | Share |
|---|---:|
| Month-to-month contract | 100% |
| Phone Service | 94% |
| No Technical Support | 92% |
| No Online Security | 91% |
| Fiber Optic Internet | 83% |
| No Online Backup | 81% |
| No Device Protection | 79% |
| Paperless Billing | 76% |
| Electronic Check Payment | 67% |

These percentages describe the prevalence of characteristics within the model-predicted churn group. They are not causal probabilities or individual churn probabilities attributable to each feature.

### Implications for Offer Reinforcement

The analysis suggests several potential areas for retention initiatives:

- Strengthening customer engagement during the early stages of the relationship.
- Evaluating retention incentives for month-to-month customers.
- Investigating the role of Online Security, Online Backup and Device Protection in customer loyalty.
- Assessing Technical Support as a customer relationship reinforcement opportunity.
- Examining whether fiber optic customers face unmet expectations or different competitive pressures.
- Designing targeted retention offers based on customer expenditure and service combinations.

These applications translate model-derived customer profiles into hypotheses for commercial and marketing intervention.

## 7. Customer Segmentation Through PCA and K-Means

The second major analytical component applies **unsupervised Machine Learning** to identify homogeneous customer segments.

While supervised classification estimates the probability of churn, clustering investigates how customers can be grouped according to shared characteristics.

The objective is to distinguish customer populations with different behavioral, contractual and economic profiles and subsequently evaluate their exposure to churn.

### Principal Component Analysis

The project applies **Principal Component Analysis (PCA)** to reduce the dimensionality of the explanatory variables before clustering.

Five principal components are retained.

**Explained variance by component:**

| Principal Component | Explained Variance |
|---|---:|
| PCA 1 | 25.95% |
| PCA 2 | 14.48% |
| PCA 3 | 8.89% |
| PCA 4 | 5.09% |
| PCA 5 | 4.17% |
| **Total** | **58.58%** |

The five components retain approximately **58.6% of the original feature variance**.

This dimensionality reduction provides a compact representation of the customer data for clustering.

### K-Means Clustering

The project applies **K-Means with K = 3** to the PCA-transformed customer data.

The clustering model is fitted to the training set and subsequently used to assign clusters to the test set.

**Cluster distribution:**

| Cluster | Training Customers | Test Customers |
|---|---:|---:|
| Cluster 0 | 1,210 | 310 |
| Cluster 1 | 2,503 | 635 |
| Cluster 2 | 1,913 | 461 |
| **Total** | **5,626** | **1,406** |

The analysis identifies three distinct customer segments.

Their churn exposure is subsequently evaluated using the observed target variable, rather than assuming that the unsupervised cluster assignments directly represent churn risk.

## 8. Statistical Validation of Customer Segments

The project evaluates whether the identified customer clusters exhibit statistically significant differences in their underlying characteristics.

Two complementary hypothesis-testing approaches are used.

### ANOVA for Continuous Variables

Analysis of Variance (ANOVA) evaluates differences among the three clusters for:

- Customer tenure.
- MonthlyCharges.
- TotalCharges.

**Results:**

| Variable | F-Statistic | Statistical Significance |
|---|---:|---|
| tenure | 1833.37 | p < 0.05 |
| MonthlyCharges | 5181.47 | p < 0.05 |
| TotalCharges | 3695.77 | p < 0.05 |

All three continuous variables show statistically significant differences among clusters.

This indicates that the segments have distinct customer relationship and expenditure characteristics.

### Chi-Square Tests for Categorical Variables

Chi-square tests evaluate whether categorical customer characteristics differ significantly across the identified clusters.

The analysis identifies statistically significant differences for multiple variables, including:

- Contract type.
- Internet service.
- Online Security and Online Backup.
- Device Protection and Technical Support.
- Payment method.
- Partner and dependents.
- Customer age category.

**Gender is an exception**, with a reported p-value of 0.1939, indicating no statistically significant difference across clusters at the 5% threshold.

These tests support the interpretation of the segments as customer groups with distinguishable characteristics.

They do not, by themselves, establish the optimal number of clusters or prove that the segmentation will remain stable on future customer populations.

## 9. Churn Risk Distribution Across Customer Clusters

The final analytical stage combines K-Means segmentation with observed churn outcomes.

For each cluster, the project calculates the proportion of customers who churned and those who remained active.

### Cluster-Level Churn Results

| Customer Segment | Churn Rate | Retention Rate | Risk Classification |
|---|---:|---:|---|
| **Cluster 0** | **8.06%** | 91.94% | Low Risk |
| **Cluster 1** | **43.94%** | 56.06% | High Risk |
| **Cluster 2** | **15.18%** | 84.82% | Moderate Relative Risk |

The three clusters exhibit substantially different churn rates.

### Cluster 0 — Stable Customers

Cluster 0 has the lowest observed churn rate, approximately **8%**.

More than 91% of customers in this group remain active.

This segment represents a relatively stable customer population.

Potential strategic priorities include loyalty reinforcement, relationship development and maintaining service satisfaction.

### Cluster 1 — High-Risk Customers

Cluster 1 has the highest observed churn rate, approximately **44%**.

This is more than five times the churn rate observed in Cluster 0.

The segment represents the most important priority for targeted churn mitigation.

Potential interventions include:

- Proactive retention campaigns.
- Contract renewal and loyalty incentives.
- Targeted offer improvements.
- Review of technical support and digital protection services.
- Personalized engagement based on customer characteristics.

### Cluster 2 — Intermediate-Risk Customers

Cluster 2 has an observed churn rate of approximately **15%**.

The segment is substantially more stable than Cluster 1 but exhibits higher churn than Cluster 0.

Potential priorities include preventive retention actions, customer engagement monitoring and selective offer reinforcement.

### Strategic Interpretation

The clustering results allow the customer base to be organized into **three differentiated churn risk segments**.

This supports a transition from uniform retention campaigns toward a more selective allocation of customer relationship management resources.

The highest-risk segment can receive greater intervention priority, while lower-risk segments can be managed through more proportionate loyalty and engagement initiatives.

## Financial and Strategic Decision-Making Applications

### Customer Retention and Revenue Protection

The integration of predictive churn modeling and customer segmentation supports the identification of customer groups with elevated attrition risk.

This can help prioritize retention activities intended to protect recurring subscription revenues.

### Marketing Budget Allocation

Different churn risk profiles can inform differentiated marketing investments.

Rather than applying uniform incentives to the entire customer base, retention budgets can be allocated according to segment risk and expected commercial relevance.

### Offer and Product Portfolio Optimization

The analysis identifies relationships between churn and specific subscription characteristics.

These insights can inform the evaluation of contract structures, service bundles and customer support offerings.

### Customer Lifetime Value Management

Customer tenure, monthly expenditure and accumulated charges emerge as important predictors.

Combining these dimensions with churn risk can provide a foundation for more advanced customer lifetime value and retention profitability models.

### Targeted Customer Relationship Management

The clustering framework can support the enrichment of customer records with segment-level churn risk indicators.

This creates a basis for differentiated retention strategies, targeted communications and more selective customer engagement initiatives.

## Key Findings

**1. Customer Churn Is Strongly Associated with Contractual and Service Characteristics**

Month-to-month contracts, limited relationship tenure and the absence of selected digital protection and support services are associated with higher churn.

**2. Economic Variables Are Important Predictive Features**

TotalCharges, tenure and MonthlyCharges emerge as the three most influential features in the Random Forest importance analysis.

**3. Predictive Model Optimization Improves Classification Performance**

Optuna optimization produces a Random Forest with approximately 80.3% accuracy and an XGBoost model with approximately 80.9% accuracy.

**4. PCA and K-Means Identify Three Customer Segments**

Five principal components explain approximately 58.6% of the original feature variance, supporting a three-cluster segmentation.

**5. Customer Segments Exhibit Statistically Significant Differences**

ANOVA and chi-square testing identify meaningful differences in expenditure, tenure and multiple contractual and service characteristics.

**6. Churn Risk Varies Substantially Across Segments**

Observed churn rates range from approximately **8% to 44%**, enabling differentiated customer retention priorities.

**7. Predictive Analytics and Clustering Provide Complementary Decision Support**

The integration of churn propensity modeling and customer segmentation connects individual risk assessment with broader marketing and customer relationship strategies.

## Technologies and Methodologies

**Programming and Data Analysis**

Python · Pandas · NumPy

**Statistical Analysis**

Pearson Correlation · Multivariate Analysis · ANOVA · Chi-Square Hypothesis Testing

**Predictive Machine Learning**

Scikit-learn · Logistic Regression · Random Forest Classification · XGBoost

**Model Optimization**

Optuna · Cross-Validation · Hyperparameter Optimization

**Unsupervised Machine Learning**

Principal Component Analysis (PCA) · K-Means Clustering

**Model Evaluation**

Accuracy · Precision · Recall · F1-Score · Confusion Matrix · Feature Importance

**Visualization**

Matplotlib · Seaborn · Correlation Heatmaps · Customer Cluster Plots

**Development Environment**

Google Colab · Jupyter Notebook

## Project Files

### Notebook

[Propensity_Model_Churn.ipynb](Propensity_Model_Churn.ipynb)

The notebook contains the complete analytical workflow, including churn driver analysis, comparative classification modeling, Optuna optimization, customer profiling, PCA, K-Means clustering, statistical validation and segment-level churn analysis.

### Dataset

The project uses the Telco Customer Churn dataset stored in this repository:

[datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv](datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv)

The notebook retrieves the dataset directly from GitHub:

https://raw.githubusercontent.com/Valeria-Raele/customer-segmentation-churn-risk/refs/heads/main/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv

Original dataset attribution: IBM Sample Data Sets.

The dataset is loaded programmatically without requiring manual uploads during notebook execution.

## Analytical Workflow

**Customer Subscription Data → Churn Driver Analysis → Statistical Feature Selection → Comparative Machine Learning → Optuna Hyperparameter Optimization → Customer Churn Profiling → PCA Dimensionality Reduction → K-Means Segmentation → ANOVA and Chi-Square Validation → Cluster-Level Churn Risk Assessment → Targeted Retention Strategy.**

The distinctive contribution of the project is the integration of supervised churn prediction and unsupervised customer segmentation within a single analytical framework.

The methodology connects the identification of churn-related offer variables with statistically differentiated customer groups, extending predictive modeling into customer retention and marketing decision support.

## Project Scope and Limitations

Independent customer analytics project focused on telecommunications churn propensity modeling, service offer analysis and customer segmentation.

The project demonstrates how predictive Machine Learning, statistical hypothesis testing and unsupervised clustering can be combined to identify customer attrition patterns and support targeted retention strategies.

The results are based on historical customer data and describe statistical relationships rather than causal effects.

Cluster-level churn rates are observed frequencies within the analyzed test sample and should not be interpreted as calibrated individual churn probabilities.

The reported classification metrics also require consideration of class imbalance: overall accuracy and weighted metrics can conceal weaker recall for the churn class.

Further validation on new customer populations, probability calibration and evaluation of intervention effectiveness would be required before operational deployment.
