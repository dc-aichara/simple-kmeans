import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


# Define Euclidean Distance function
def dist(p1, p2):
    d = (p1 - p2) ** 2
    distance = np.sqrt(d[0] + d[1])
    return distance


# Define K-means clustering function
def k_means(k, df_points, max_iters):
    df = pd.read_csv(df_points)
    df.columns = ['x', 'y']
    points = df.values

    # Initiate with first k points as centroids
    centroids = []
    for i in range(0, k):
        centroids.append(df.values[i])
    # calculate distance of each point from centroid
    # Iterate through length of max iterations
    for n in range(0, max_iters):

        for i in range(0, k):
            dis = []
            for point in points:
                dis.append(dist(point, centroids[i]))
            df['distance_{}'.format(i)] = dis

        # Define classes bases on distance between points and centroids
        classes = []
        for j in range(len(df)):
            c = df[df.columns.tolist()[2:5]].values[j].tolist()
            classes.append(c.index(min(c)))
        df['classes'] = classes

        # Find new Centroids (Calculate mean points of each class)
        for m in range(0, k):
            df1 = df[df['classes'] == m]
            x = df1[df1.columns.tolist()[:2]].values
            centroids[m] = x.mean(axis=0)

    centroids = pd.DataFrame(centroids, columns=['x', 'y'])

    # Save Clusters and Centroids
    df = df[['x', 'y', 'classes']]
    df.to_csv('results/cluster.csv', index=False)
    centroids.to_csv('results/centroids.csv', index=False)

    # Let's Plot the results
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.scatterplot('x', 'y', s=100, hue='classes', style='classes', data=df)
    sns.scatterplot('x', 'y', s=250, marker='o', color='b', data=centroids)
    i = 0
    for x, y in centroids.values:
        plt.text(x, y, 'Centroid {}'.format(i), color='yellow', fontsize=20)
        i += 1
    plt.title('Simple K-means Clustering', color='b', fontsize=27)
    plt.xlabel('X', color='b', fontsize=23)
    plt.ylabel('Y', color='b', fontsize=23)
    fig.set_facecolor('y')
    ax.set_facecolor('gray')
    ax.grid()
    plt.savefig('results/kmeans_clustering.png', bbox_inches='tight', facecolor='yellow')

    return df, centroids


def main():
    k_means(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]))


if __name__ == '__main__':
    main()



