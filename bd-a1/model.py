import pandas as pd
from sklearn.cluster import KMeans
import sys
import subprocess

def k_means_algorithm(df):

    # df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Embarked'])
    # Example: selecting columns suitable for K-means
    X = df[['Pclass', 'Survived']]  

    # Initialize K-means model
    kmeans = KMeans(n_clusters=3)

    # Fit the model
    kmeans.fit(X)

    # Get the number of records in each cluster
    cluster_counts = pd.Series(kmeans.labels_).value_counts().sort_index()

    # Save the number of records in each cluster
    with open('k.txt', 'w') as f:
        f.write(str(cluster_counts))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python model.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    df = pd.read_csv(file_path)

    k_means_algorithm(df)

    print("K-means algorithm executed successfully.")

    subprocess.run(["python3", "vis.py", file_path])