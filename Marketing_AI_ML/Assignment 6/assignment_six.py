import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
import matplotlib.pyplot as plt
import os 
import seaborn as sns; sns.set(style="ticks", color_codes=True)

current_dir = os.getcwd()

file_name = "Mall_Customers.csv"
df = pd.read_csv(file_name)
data = df.drop(columns=['Genre'])

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data)
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=15, random_state=0)
kmeans.fit(scaled_features)

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

    data_with_clusters['Cluster Values'] = data_with_clusters['Clusters'].replace([0,1,2,3], [c0_range, c1_range, c2_range, c3_range])
    sns.boxplot(x=data_with_clusters['Cluster Values'], y=data_with_clusters['Age'], palette="Set3", hue=data_with_clusters['Clusters'], dodge=False)
    plt.legend(loc=2, bbox_to_anchor= (1,1), title='Clusters Group ID')
    plt.tight_layout()
    plt.savefig(current_dir +'/media/age_chart.png')
    plt.clf()


def Annual_Income_chart():
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

    data_with_clusters['Cluster Values'] = data_with_clusters['Clusters'].replace([0,1,2,3], [c0_range, c1_range, c2_range, c3_range])
    sns.boxplot(x=data_with_clusters['Cluster Values'], y=data_with_clusters['Annual Income (k$)'], palette="Set3", hue=data_with_clusters['Clusters'], dodge=False)
    plt.legend(loc=2, bbox_to_anchor= (1,1), title='Clusters Group ID')
    plt.tight_layout()
    plt.savefig(current_dir +'/media/annual_chart.png')
    plt.clf()


def Spending_Score_chart():
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

    data_with_clusters['Cluster Values'] = data_with_clusters['Clusters'].replace([0,1,2,3], [c0_range, c1_range, c2_range, c3_range])
    sns.boxplot(x=data_with_clusters['Cluster Values'], y=data_with_clusters['Spending Score (1-100)'], palette="Set3", hue=data_with_clusters['Clusters'], dodge=False)
    plt.legend(loc=2, bbox_to_anchor= (1,1), title='Clusters Group ID')
    plt.tight_layout()
    plt.savefig(current_dir +'/media/Spending_Score_chart.png')
    plt.clf()

age_chart()
Annual_Income_chart()
Spending_Score_chart()
elbow_chart()