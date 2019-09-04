import matplotlib.pyplot as plt
import pandas as pd
import numpy as np      
import scipy.cluster.hierarchy as sch
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans


def plot_top10_dist(data, city, color_plot): 
    # Top 10 itemIDs delivered
    plt.figure(figsize = (13,3))
    plt.subplot(121)
    sns.countplot(x = 'item_id', data = data
                  , order = data['item_id'].value_counts().iloc[:10].index
                  , color = color_plot)
    
    plt.xticks(rotation=90)
    plt.xlabel('Item IDs',fontsize = 12)
    plt.ylabel('Number of Deliveries \n',fontsize = 12)
    plt.title('Top 10 Items Delivered: ' + city,fontsize = 15);

    # Histogram of number of times items were delivered
    plt.subplot(122)
    sns.distplot(data['item_id'].value_counts(), kde=False, color = color_plot);
    plt.xlabel('Number of Deliveries',fontsize = 12)
    plt.ylabel('Counts',fontsize = 12);
    plt.title('Distribution of Order Frequencies (Total)',fontsize = 15);


def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    
def OrderID_to_ItemID(data):
    # Create dataframe of items bought for every order ID
    basket = (data.groupby(['order_id', 'item_id'])['itemqty']
              .sum().unstack().reset_index().fillna(0)
              .set_index('order_id'))

    # Convert column/item names as strings
    basket.columns = basket.columns.astype(str)
    
    # One hot-encode items instead of quanitity purchased 
    basket_sets = basket.applymap(encode_units)
    return basket_sets


def PCA_feature_select(X, retain, city):
    # Create PCA instance
    pca = PCA()
    principalComponents = pca.fit_transform(X)
    
    # Plot explained variances
    features = range(pca.n_components_)
    plt.figure(figsize = (20,5))
    plt.subplot(131)
    plt.bar(features, pca.explained_variance_ratio_, color='black')
    plt.xlabel('PCA Features', fontsize = 15)
    plt.ylabel('Proportion of Variance Explained', fontsize = 15)
    plt.title('PCA: ' + city, fontsize = 20)
    plt.xticks(features)
    plt.xlim(-0.5,10.5)

    # Save components to a DataFrame
    PCA_components = pd.DataFrame(principalComponents)
    
    # Compute and plot cumulative variance of principal components
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    plt.subplot(132)
    plt.plot(range(0,len(cumulative)), cumulative, color = 'black');
    plt.scatter(range(0,len(cumulative)), cumulative, color = 'black');
    plt.xlabel('PCA Features', fontsize = 15)
    plt.ylabel('Explained Variance', fontsize = 15)
    plt.title('Cumulative Explained Variance', fontsize = 20)
    plt.grid(True)
    
    # Extracting the PCA components with at least 75% of cumulative variance
    
    keep = min(range(len(cumulative)), key=lambda i: abs(cumulative[i]-retain))
    Xreduced = PCA_components.iloc[:,:keep+1];

    # Scatter plot of the first two components of PCA model
    plt.subplot(133)
    plt.scatter(PCA_components[0], PCA_components[1], alpha=.1, color='black')
    plt.xlabel('PCA 1', fontsize = 15)
    plt.ylabel('PCA 2', fontsize = 15)
    plt.title('Principal Component Plot', fontsize = 20)
    plt.show();
    return Xreduced


def Kmeans_no_clusters(Xreduced, max_number_clusters, city):
    # Initialize range of k to test
    ks = range(1, max_number_clusters)

    # Implement the elbow method 
    inertias = []
    for k in ks:
        model = KMeans(n_clusters = k)
        model.fit(Xreduced)
        inertias.append(model.inertia_)

    # Plotting changes in inertia as a function of cluster    
    plt.figure()
    plt.plot(ks, inertias, '-o', color ='black')
    plt.xlabel('Number of Clusters', fontsize = 15)
    plt.ylabel('Inertia \n', fontsize = 15)
    plt.title('\n K-means Elbow Method: ' + city, fontsize = 20)
    plt.xticks(ks)
    plt.grid(True)
    plt.show()
    
def dendrogram_visual(X, city):
    # Dendrogram visualization
    plt.figure(figsize = (18,5))
    plt.title('Dendrogram: ' + city, fontsize = 20)
    plt.xlabel('\n Item IDs', fontsize = 15)
    plt.ylabel('Distance \n', fontsize = 15)

    dendrogram = sch.dendrogram(sch.linkage(X, method='ward')
                                , above_threshold_color='black'
                                , leaf_rotation = 90
                                , leaf_font_size = 11
                                , labels = X.index)
    plt.show()
    

def Agglomerative_Cluster(X, clusters, city):
    # Perform Agglomerative Clustering (number of clusters based on plots above)
    cluster_no = AgglomerativeClustering(n_clusters=clusters, affinity='euclidean', linkage='ward')  
    X['Labels'] = cluster_no.fit_predict(X)
    
    # Visualize clusters in PC space
    fig = plt.figure(figsize = (5,5))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('Hierarchical Clustering: ' + city, fontsize = 20)
    targets = range(0, clusters+1)
    colors = ['g', 'b', 'r', 'c', 'm']
    for target, color in zip(targets, colors):
        indicesToKeep = X['Labels'] == target
        ax.scatter(X.loc[indicesToKeep, 0]
                   , X.loc[indicesToKeep, 1]
                   , c = color
                   , s = 50
                   , alpha = 0.1)
    ax.grid()
    
    return X

