

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. LOAD DATA
df = pd.read_excel('professor_review_project_dataset.xlsx')
print(df.shape)
print(df.info())
print(df.isna().sum())

# 2. STANDARDIZE CATEGORICAL VALUES
for col in ['Department','Attendance_Required','Would_Take_Again','Grade_Received']:
    df[col] = df[col].astype('string').str.strip()
df['Department'] = df['Department'].replace({
    'Computer science':'Computer Science', 'MATH':'Mathematics'})
df['Attendance_Required'] = df['Attendance_Required'].str.title().replace(
    {'N':'No','Required':'Yes'})
df['Would_Take_Again'] = df['Would_Take_Again'].str.title()

# 3. RANGE VALIDATION: INVALID -> MISSING
rules = {
    'Rating':(0,5), 'Difficulty':(1,5),
    'Professor_Experience_Years':(0,60),
    'Class_Size':(1,300), 'Study_Hours_Per_Week':(0,40)}
for col,(low,high) in rules.items():
    bad = ~df[col].between(low,high) & df[col].notna()
    df.loc[bad,col] = np.nan

# 4. DEDUPLICATE + IMPUTE
df = df.drop_duplicates(subset=['Review_ID'], keep='first')
df.loc[df['Would_Take_Again']=='Maybe','Would_Take_Again'] = pd.NA
numeric = ['Rating','Difficulty','Professor_Experience_Years',
           'Class_Size','Study_Hours_Per_Week']
for col in numeric:
    df[col] = df[col].fillna(df[col].median())
df['Grade_Received'] = df['Grade_Received'].fillna('Unknown')
df['Would_Take_Again'] = df['Would_Take_Again'].fillna(
    df['Would_Take_Again'].mode()[0])

# 5. FEATURE ENGINEERING
df['Take_Again_Binary'] = df['Would_Take_Again'].map({'Yes':1,'No':0})
df['High_Rating'] = (df['Rating'] >= 4).astype(int)
df['Workload_Index'] = df['Difficulty'] * df['Study_Hours_Per_Week']

# 6. EDA
print(df[numeric].describe())
print(df.groupby('Department')['Rating'].mean())
print(df[numeric].corr())
plt.hist(df['Rating'], bins=10)
plt.xlabel('Rating'); plt.ylabel('Frequency'); plt.show()
plt.scatter(df['Difficulty'],df['Rating'])
plt.xlabel('Difficulty'); plt.ylabel('Rating'); plt.show()

# 7. HYPOTHESIS TESTS
r,p = stats.pearsonr(df['Difficulty'],df['Rating'])
print('Pearson:',r,p)
yes = df.loc[df.Would_Take_Again=='Yes','Rating']
no = df.loc[df.Would_Take_Again=='No','Rating']
print('Welch t-test:', stats.ttest_ind(yes,no,equal_var=False))
groups=[g.Rating.values for _,g in df.groupby('Department')]
print('ANOVA:',stats.f_oneway(*groups))
ct=pd.crosstab(df.Department,df.Would_Take_Again)
print('Chi-square:',stats.chi2_contingency(ct))

# 8. PCA
pca_features=['Rating','Difficulty','Professor_Experience_Years',
              'Class_Size','Study_Hours_Per_Week']
Z=StandardScaler().fit_transform(df[pca_features])
pca=PCA().fit(Z)
PC=pca.transform(Z)
print('PCA eigenvalues:',pca.explained_variance_)
print('PCA explained variance ratio:',pca.explained_variance_ratio_)

# 9. LDA
Xlda=StandardScaler().fit_transform(df[pca_features])
ylda=df['Take_Again_Binary']
lda=LinearDiscriminantAnalysis(n_components=1)
L=lda.fit_transform(Xlda,ylda)
print('LDA training accuracy:',lda.score(Xlda,ylda))

# 10. SIMPLE LINEAR REGRESSION
X=df[['Difficulty']]; y=df['Rating']
slr=LinearRegression().fit(X,y)
pred=slr.predict(X)
print('SLR equation:',slr.intercept_,slr.coef_[0])
print('R2:',r2_score(y,pred))
print('MAE:',mean_absolute_error(y,pred))
print('RMSE:',mean_squared_error(y,pred)**0.5)

# 11. MULTIPLE LINEAR REGRESSION
features=['Difficulty','Professor_Experience_Years','Class_Size',
          'Study_Hours_Per_Week','Student_Year']
X=df[features]
mlr=LinearRegression().fit(X,y)
pred=mlr.predict(X)
r2=r2_score(y,pred); n=len(y); p=len(features)
adj_r2=1-(1-r2)*(n-1)/(n-p-1)
print('MLR coefficients:',dict(zip(features,mlr.coef_)))
print('R2 / Adjusted R2:',r2,adj_r2)

# 12. TRAIN / TEST
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42)
mlr.fit(Xtr,ytr); test_pred=mlr.predict(Xte)
print('Test R2:',r2_score(yte,test_pred))
print('Test MAE:',mean_absolute_error(yte,test_pred))
print('Test RMSE:',mean_squared_error(yte,test_pred)**0.5)




