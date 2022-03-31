import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
import os 
import seaborn as sns; sns.set(style="ticks", color_codes=True)
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

def elbow_chart():
    visualizer = KElbowVisualizer(KMeans(), k=(1,15)).fit(scaled_features)
    plt.legend(bbox_to_anchor= (1,1))
    plt.savefig(current_dir +'/media/elbow_chart.png')
    plt.clf()


def age_chart():
    #building the cluster labels
    c0_min = min(c0['Age'])
    c0_max = max(c0['Age'])
    c0_range = ('{} - {}'.format(c0_min, c0_max))

    c1_min = min(c1['Age'])
    c1_max = max(c1['Age'])
    c1_range = ('{} - {}'.format(c1_min, c1_max))

    c2_min = min(c2['Age'])
    c2_max = max(c2['Age'])
    c2_range = ('{} - {}'.format(c2_min, c2_max))

    c3_min = min(c3['Age'])
    c3_max = max(c3['Age'])
    c3_range = ('{} - {}'.format(c3_min, c3_max))

    # the labels would normally have the cluster IDs. i'm replacing them with the cluster ranges above
    data_with_clusters['Cluster Values'] = data_with_clusters['Clusters'].replace([0,1,2,3], [c0_range, c1_range, c2_range, c3_range])
    sns.boxplot(x=data_with_clusters['Cluster Values'], y=data_with_clusters['Age'], palette="Set3", hue=data_with_clusters['Clusters'], dodge=False)
    plt.legend(loc=2, bbox_to_anchor= (1,1), title='Clusters Group ID')
    plt.tight_layout()
    plt.savefig(current_dir +'/media/age_chart.png')
    plt.clf()


def Annual_Income_chart():
    #building the cluster labels
    c0_min = min(c0['Annual Income (k$)'])
    c0_max = max(c0['Annual Income (k$)'])
    c0_range = ('{} - {}'.format(c0_min, c0_max))

    c1_min = min(c1['Annual Income (k$)'])
    c1_max = max(c1['Annual Income (k$)'])
    c1_range = ('{} - {}'.format(c1_min, c1_max))

    c2_min = min(c2['Annual Income (k$)'])
    c2_max = max(c2['Annual Income (k$)'])
    c2_range = ('{} - {}'.format(c2_min, c2_max))

    c3_min = min(c3['Annual Income (k$)'])
    c3_max = max(c3['Annual Income (k$)'])
    c3_range = ('{} - {}'.format(c3_min, c3_max))

    # the labels would normally have the cluster IDs. i'm replacing them with the cluster ranges above
    data_with_clusters['Cluster Values'] = data_with_clusters['Clusters'].replace([0,1,2,3], [c0_range, c1_range, c2_range, c3_range])
    sns.boxplot(x=data_with_clusters['Cluster Values'], y=data_with_clusters['Annual Income (k$)'], palette="Set3", hue=data_with_clusters['Clusters'], dodge=False)
    plt.legend(loc=2, bbox_to_anchor= (1,1), title='Clusters Group ID')
    plt.tight_layout()
    plt.savefig(current_dir +'/media/annual_chart.png')
    plt.clf()


def Spending_Score_chart():

    #building the cluster labels
    c0_min = min(c0['Spending Score (1-100)'])
    c0_max = max(c0['Spending Score (1-100)'])
    c0_range = ('{} - {}'.format(c0_min, c0_max))

    c1_min = min(c1['Spending Score (1-100)'])
    c1_max = max(c1['Spending Score (1-100)'])
    c1_range = ('{} - {}'.format(c1_min, c1_max))

    c2_min = min(c2['Spending Score (1-100)'])
    c2_max = max(c2['Spending Score (1-100)'])
    c2_range = ('{} - {}'.format(c2_min, c2_max))

    c3_min = min(c3['Spending Score (1-100)'])
    c3_max = max(c3['Spending Score (1-100)'])
    c3_range = ('{} - {}'.format(c3_min, c3_max))

    # the labels would normally have the cluster IDs. i'm replacing them with the cluster ranges above
    data_with_clusters['Cluster Values'] = data_with_clusters['Clusters'].replace([0,1,2,3], [c0_range, c1_range, c2_range, c3_range])
    sns.boxplot(x=data_with_clusters['Cluster Values'], y=data_with_clusters['Spending Score (1-100)'], palette="Set3", hue=data_with_clusters['Clusters'], dodge=False)
    plt.legend(loc=2, bbox_to_anchor= (1,1), title='Clusters Group ID')
    plt.tight_layout()
    plt.savefig(current_dir +'/media/Spending_Score_chart.png')
    plt.clf()


def overall_data():
    #since for the project I selected Cluster 3 for analysis, i'm doing analysis for that cluster.

    #ols can't handle non-alphanumeric characters
    c3.rename(columns={'Spending Score (1-100)': "Spend", 'Annual Income (k$)':'Income'}, inplace=True)
    lm = ols('Spend ~ Gender + Age + Income', data=c3).fit()
    print('Male & Female stats')
    print(lm.summary())

# 1 is male and 2 is female
def gender_data(gender_id):
    #depending on the gender selected for analysis, the label is updated
    gender_name = ''
    for key, value in gender_cat.items():
        if gender_id == value:
            gender_name = key

    #only selecting values that match the gender ID from the cluster selected
    df = c3.loc[c3['Gender'] == gender_id]
    #linear regression plots
    sns.lmplot(x='Annual Income (k$)', y='Spending Score (1-100)', data = df)
    sns.lmplot(x='Age', y='Spending Score (1-100)', data = df)
    sns.lmplot(x='Age', y='Annual Income (k$)', data = df)

    x = df['Age']
    y = df['Annual Income (k$)']
    z = df['Spending Score (1-100)']

    #performing different correlation analysis 
    income_spend_corr = y.corr(z)
    age_spend_corr = x.corr(z)
    age_income_corr = x.corr(y)
    correlation = gender_name+' Correlation: \nincome_spending correlation: {} \nage_spending correlation: {}\nage_income correlation: {}\n'.format(income_spend_corr, age_spend_corr, age_income_corr)
    print(correlation)  

    #performing different means of each attribute
    age_mean = df['Age'].mean()
    income_mean = df['Annual Income (k$)'].mean()
    spend_mean = df['Spending Score (1-100)'].mean()
    means = gender_name+' means: \nage mean: {} \nspend mean: {} \nincome mean: {}'.format(age_mean, spend_mean, income_mean)
    print(means)

    #ANOVA 
    df.rename(columns={'Spending Score (1-100)': "Spend", 'Annual Income (k$)':'Income'}, inplace=True)
    lm = ols('Spend ~ Age + Income', data=df).fit()
    table = sm.stats.anova_lm(lm)
    print(lm.summary())

if __name__ == "__main__":
    # age_chart()
    # Annual_Income_chart()
    # Spending_Score_chart()
    # elbow_chart()
    # gender_data(1)
    # gender_data(2)
    overall_data()