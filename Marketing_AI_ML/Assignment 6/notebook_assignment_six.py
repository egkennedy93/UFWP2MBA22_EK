
# %%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os 
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from scipy.stats import ttest_1samp, f_oneway, ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols

current_dir = os.getcwd()

# defining the file and dropping genre since pandas doesn't like strings
file_name = "Mall_Customers.csv"
df = pd.read_csv(file_name).rename(columns={'Genre': 'Gender'})
gender_cat = {'Male': 1, 'Female': 2}
# data = df.drop(columns=['Genre'])
df['Gender'] = df['Gender'].map(gender_cat)
data = df

#using a scaler to standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data)

#After looking at the elbow_chart function below, determined n_clusters to be 4. Now fitting the scaled_features to
# a 4 seed cluster
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=15, random_state=0)
kmeans.fit(scaled_features)

#copying the fitted data to a new df, and adding the clusters column 
data_with_clusters = data.copy()
data_with_clusters['Clusters'] = kmeans.labels_

# defining clusters into 4 sepearte DFs
c0 = data_with_clusters.loc[data_with_clusters['Clusters'] == 0]
c1 = data_with_clusters.loc[data_with_clusters['Clusters'] == 1]
c2 = data_with_clusters.loc[data_with_clusters['Clusters'] == 2]
c3 = data_with_clusters.loc[data_with_clusters['Clusters'] == 3]
c3
#%%
sns.lmplot(x='Annual Income (k$)', y='Spending Score (1-100)', data = c3)
sns.lmplot(x='Age', y='Spending Score (1-100)', data = c3)
sns.lmplot(x='Age', y='Annual Income (k$)', data = c3)

x = c3['Age']
y = c3['Annual Income (k$)']
z = c3['Spending Score (1-100)']

age_income_corr = x.corr(y)
age_spend_corr = x.corr(z)
income_spend_corr = y.corr(z)
#%%
male_df = c3.loc[c3['Gender'] == 1]
male_df

sns.lmplot(x='Annual Income (k$)', y='Spending Score (1-100)', data = male_df)
sns.lmplot(x='Age', y='Spending Score (1-100)', data = male_df)
sns.lmplot(x='Age', y='Annual Income (k$)', data = male_df)

m_x = male_df['Age']
m_y = male_df['Annual Income (k$)']
m_z = male_df['Spending Score (1-100)']

m_income_spend_corr = y.corr(z)
m_age_spend_corr = x.corr(z)
m_age_income_corr = x.corr(y)

m_age_mean = male_df['Age'].mean()
m_spend_mean = male_df['Annual Income (k$)'].mean()
m_income_mean = male_df['Spending Score (1-100)'].mean()

# %%
female_df = c3.loc[c3['Gender'] == 2]
sns.lmplot(x='Annual Income (k$)', y='Spending Score (1-100)', data = female_df)
sns.lmplot(x='Age', y='Spending Score (1-100)', data = female_df)
sns.lmplot(x='Age', y='Annual Income (k$)', data = female_df)

f_x = female_df['Age']
f_y = female_df['Annual Income (k$)']
f_z = female_df['Spending Score (1-100)']

f_income_spend_corr = y.corr(z)
f_age_spend_corr = x.corr(z)
f_age_income_corr = x.corr(y)

f_age_mean = male_df['Age'].mean()
f_spend_mean = male_df['Annual Income (k$)'].mean()
f_income_mean = male_df['Spending Score (1-100)'].mean()

 
#%%
test = ttest_1samp(female_df['Age'], b=female_df['Spending Score (1-100)'].mean(), equal_var=True)
test
# %%
df = data.rename(columns={'Spending Score (1-100)': "Spend", 'Annual Income (k$)':'Income'}, inplace=True)
# %%
lm = ols('Spend ~ Gender + Age + Income', data=c3).fit()
table = sm.stats.anova_lm(lm)
# %%
print(lm.summary())
# %%
male_df.rename(columns={'Spending Score (1-100)': "Spend", 'Annual Income (k$)':'Income'}, inplace=True)
lm = ols('Spend ~ Age + Income', data=male_df).fit()
table = sm.stats.anova_lm(lm)
print(lm.summary())
# %%
female_df.rename(columns={'Spending Score (1-100)': "Spend", 'Annual Income (k$)':'Income'}, inplace=True)
lm = ols('Spend ~ Age + Income', data=female_df).fit()
table = sm.stats.anova_lm(lm)
print(lm.summary())
# %%
