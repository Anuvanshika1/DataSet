# Professor Review Analysis Using Data Science

## Fundamentals of Data Science Project

This project performs an **exploratory, statistical, and predictive analysis of professor review data** using concepts studied in the **Fundamentals of Data Science (FODS)** course.

The objective of this project is not to build a professor-review product or application. Instead, it applies the **data science workflow** to understand the dataset, identify data-quality problems, discover patterns, test statistical relationships, perform feature engineering and dimensionality reduction, and study factors associated with professor ratings.

---

## Project Objectives

The main objectives are to:

- Understand the structure and characteristics of professor review data.
- Identify numerical and categorical variables.
- Study the four types of data analytics:
  - Descriptive Analytics
  - Diagnostic Analytics
  - Predictive Analytics
  - Prescriptive Analytics
- Assess and improve data quality.
- Handle missing, invalid, inconsistent, and duplicate data.
- Perform exploratory data analysis.
- Calculate descriptive statistics.
- Study relationships between variables.
- Perform hypothesis testing.
- Apply feature engineering techniques.
- Perform dimensionality reduction using PCA.
- Perform supervised dimensionality reduction using LDA.
- Study the relationship between professor difficulty and rating.
- Build and interpret a Simple Linear Regression model.
- Explore Multiple Linear Regression.
- Evaluate the limitations of the dataset and suggest improvements.

---

## Dataset

The project uses a **Professor Review Dataset** containing student feedback and professor/course-related attributes.

Important variables include:

| Feature | Description |
|---|---|
| `Review_ID` | Unique identifier for a review |
| `Department` | Professor/course department |
| `Course_Type` | Type of course |
| `Rating` | Student rating |
| `Difficulty` | Perceived course/professor difficulty |
| `Years_Experience` | Teaching experience |
| `Avg_Study_Hours` | Average study hours |
| `Class_Size` | Number of students in the class |
| `Would_Take_Again` | Whether the student would take the professor again |

The original dataset intentionally contains data-quality problems such as missing values, inconsistent category labels, duplicate identifiers, and invalid numerical values.

---

# Dataset Version 3

The project also introduces **Version 3 of the dataset**.

Version 3 adds information required to evaluate data-quality dimensions that could not be measured properly using the original dataset.

Additional fields include:

- `Review_Date`
- `Last_Updated`
- `Professor_ID`
- `Course_Master_ID`
- `Verified_Source`
- `Verified_Rating`

It also contains supporting master data for professors and courses.

These additions allow the project to demonstrate:

- Accuracy
- Timeliness
- Referential Integrity

The Version 3 fields are project extensions created to demonstrate a more complete data-quality framework and should not be interpreted as original institutional records.

---

# Four Types of Data Analytics

## 1. Descriptive Analytics

**Question:** What happened?

Used to summarize the professor review dataset using:

- Mean
- Median
- Mode
- Minimum
- Maximum
- Variance
- Standard deviation
- Frequency distributions
- Histograms
- Bar charts

---

## 2. Diagnostic Analytics

**Question:** Why did it happen?

Used to investigate relationships between variables.

Examples include:

- Difficulty vs Rating
- Department vs Rating
- Would Take Again vs Rating
- Pearson Correlation
- Hypothesis Testing
- Chi-Square Test
- ANOVA

---

## 3. Predictive Analytics

**Question:** What is likely to happen?

Regression analysis is used to investigate whether professor ratings can be estimated using available features.

Techniques include:

- Simple Linear Regression
- Multiple Linear Regression

---

## 4. Prescriptive Analytics

**Question:** What should be done?

Prescriptive analytics is comparatively limited in the original dataset because the dataset mainly contains historical review observations.

It can be improved by collecting additional information such as:

- teaching interventions,
- recommended actions,
- action dates,
- follow-up ratings,
- student outcomes before and after interventions,
- teaching-method changes.

This would allow future analysis to recommend actions and evaluate whether those actions actually improve student experience.

---

# Data Quality Assessment

The project studies seven major dimensions of data quality:

1. Accuracy
2. Completeness
3. Consistency
4. Validity
5. Uniqueness
6. Timeliness
7. Referential Integrity

Examples of detected problems include:

- Missing values
- Duplicate review IDs
- Inconsistent category capitalization/spelling
- Ratings outside the valid range
- Invalid difficulty values
- Negative experience values
- Negative study-hour values
- Unusually large class-size values

The dataset is cleaned before performing statistical and machine-learning analysis.

---

# Data Cleaning

Cleaning operations include:

- Handling missing values
- Removing duplicate records
- Standardizing categorical values
- Checking numerical ranges
- Correcting or treating invalid observations
- Converting columns to suitable data types
- Preparing numerical features for analysis

---

# Exploratory Data Analysis

The project performs exploratory analysis using visualizations such as:

- Histograms
- Bar charts
- Box plots
- Scatter plots

A scatter plot is used particularly to study the relationship between:

```text
Difficulty → Rating
```

---

# Descriptive Statistics

Statistics are calculated to understand the central tendency and spread of the data.

## Mean

```text
Mean = Sum of observations / Number of observations
```

## Median

The observations are sorted and the middle observation is selected.

For an even number of observations, the average of the two middle observations is used.

## Mode

The value occurring most frequently in the dataset is selected.

## Variance and Standard Deviation

These measure how much the observations vary around their mean.

Where possible, the project demonstrates these calculations manually instead of relying only on built-in statistical functions.

---

# Correlation Analysis

Pearson correlation is used to study the strength and direction of the linear relationship between numerical variables.

One of the primary relationships investigated is:

```text
Difficulty vs Rating
```

The analysis indicates a negative relationship between these variables in the project dataset.

This means that higher difficulty tends to be associated with lower ratings in the observed data.

Correlation represents **association and not causation**.

---

# Hypothesis Testing

Statistical hypothesis tests are used to determine whether observed differences or relationships are statistically meaningful.

The project includes:

## Welch's t-test

Used to compare average ratings between students who answered:

```text
Would_Take_Again = Yes
```

and

```text
Would_Take_Again = No
```

### Null Hypothesis

There is no significant difference between the average ratings of the two groups.

### Alternative Hypothesis

There is a significant difference between the average ratings of the two groups.

---

## Chi-Square Test

Used to study relationships between categorical variables.

The calculation is demonstrated using observed and expected frequencies.

---

## ANOVA

ANOVA can be used when average ratings need to be compared across more than two groups, such as multiple departments.

---

# Feature Engineering

Feature engineering prepares useful variables for further analysis.

Techniques explored include:

- Feature extraction
- Feature construction
- Feature selection
- Encoding categorical variables
- Standardization
- Feature reduction

Categorical values can be transformed into numerical representations using techniques such as one-hot encoding.

---

# Principal Component Analysis — PCA

PCA is an **unsupervised dimensionality-reduction technique**.

Its objective is to transform correlated numerical features into new principal components while preserving as much variance as possible.

The project follows these steps:

1. Select numerical features.
2. Handle missing values.
3. Standardize the data.
4. Calculate the covariance matrix.
5. Calculate eigenvalues.
6. Calculate eigenvectors.
7. Sort components according to eigenvalues.
8. Calculate explained variance.
9. Select important principal components.
10. Project the original data onto the selected components.

PCA does not use the target/class label.

---

# Linear Discriminant Analysis — LDA

LDA is a **supervised dimensionality-reduction technique**.

Unlike PCA, LDA uses class labels.

In this project:

```text
Target = Would_Take_Again
```

The main classes are:

```text
Yes
No
```

LDA attempts to:

- minimize variation within each class,
- maximize separation between classes.

The process includes:

1. Separating observations by class.
2. Calculating class means.
3. Calculating the overall mean.
4. Calculating the within-class scatter matrix.
5. Calculating the between-class scatter matrix.
6. Finding eigenvalues and eigenvectors.
7. Selecting the best discriminant direction.
8. Projecting observations onto that direction.

---

# Simple Linear Regression

Simple Linear Regression is used to investigate the relationship between:

```text
Independent Variable (X) = Difficulty
Dependent Variable (Y)   = Rating
```

The general regression equation is:

```text
Predicted Rating = Intercept + Slope × Difficulty
```

The project calculates:

- Mean of X
- Mean of Y
- Deviations
- Products of deviations
- Squared deviations
- Slope
- Intercept
- Predicted values
- Residuals
- SSE
- SST
- R²
- MAE
- M
