# 🧪 Schistosoma mansoni in Uganda: Prevalence, Distribution & Determinants

This repository contains an R-based analysis of *Schistosoma mansoni* infections in Uganda. Using WHO-AFRO region survey data, this project summarizes prevalence trends, spatial distribution, and key predictors, supporting evidence-based decision-making in public health.

## 📈 Summary of Findings

### 1. 🗂️ Data Summary
- **Total surveys conducted**: 2,205  
- **Individuals tested**: 181,037  
- **Positive S. mansoni cases**: 38,302  
- **Data completeness**: 82.75%

### 2. 📊 Prevalence Over Time
- Visualized yearly mean prevalence with 95% confidence intervals.
- Added yearly population estimates for school-aged children.
- **Welch Two Sample t-test** (pre vs. post-2000):
  - Mean before 2000: **18.98%**
  - Mean after 2000: **19.98%**
  - **p = 0.506** (not statistically significant)

📌 *Conclusion*: No statistically significant change in mean prevalence before vs. after 2000.

[Mean Prevalence with school aged children.pdf](https://github.com/user-attachments/files/21015552/Mean.Prevalence.with.school.aged.children.pdf)

### 3. 🗺️ Regional Prevalence & Uncertainty

| Region      | Mean Prevalence (%) | 95% CI           | Sample Size |
|-------------|---------------------|------------------|-------------|
| Northern    | 24.1                | (22.0, 26.3)     | 619         |
| Eastern     | 19.4                | (17.9, 20.9)     | 991         |
| Western     | 18.0                | (15.5, 20.4)     | 498         |
| Central     | 16.2                | (14.7, 17.7)     | 674         |
| NA          | 39.5                | (30.5, 48.5)     | 60          |
| Nord Kivu   | 0.0                 | (0.0, 0.0)       | 2           |

- **Northern Region** shows the highest well-supported prevalence.
- **Central Region** shows the lowest among major regions.

### 4. 🤖 Determinants of Prevalence (Modeling)
Four models were trained using an 80/20 train-test split.  
**Best model**: `Random Forest` (lowest RMSE).

| Model             | RMSE   |
|------------------|--------|
| Random Forest     | 19.71  |
| XGBoost           | 20.67  |
| Linear Regression | 26.93  |
| Lasso Regression  | 28.28  |

🔑 **Key predictors**:
- Latitude / Longitude  
- Survey Year  
- Participant age range

📌 *Model Justification*: Random Forest best captured non-linearities and interactions in the dataset.


