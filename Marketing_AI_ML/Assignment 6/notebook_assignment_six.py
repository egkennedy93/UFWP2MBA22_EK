
# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns

file_name = "Mall_Customers.csv"
df = pd.read_csv(file_name)

data = df.drop(columns=['Genre'])

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data)

visualizer = KElbowVisualizer(KMeans(), k=(1,15)).fit(scaled_features)
plt.legend(loc=2, bbox_to_anchor= (1,1))
visualizer.show()
#%%
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=15, random_state=0)
kmeans.fit(scaled_features)

scaled_features

data_with_clusters = data.copy()
data_with_clusters['Clusters'] = kmeans.labels_
#%%
c0 = data_with_clusters.loc[data_with_clusters['Clusters'] == 0]
c1 = data_with_clusters.loc[data_with_clusters['Clusters'] == 1]
c2 = data_with_clusters.loc[data_with_clusters['Clusters'] == 2]
c3 = data_with_clusters.loc[data_with_clusters['Clusters'] == 3]

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

data_with_clusters['Clusters'] = data_with_clusters['Clusters'].replace([0,1,2,3], [c0_range, c1_range, c2_range, c3_range])
box = sns.boxplot(x=data_with_clusters['Clusters'], y=data_with_clusters['Age'], palette="Set3")
# %%
kmeans.labels_
# %%
