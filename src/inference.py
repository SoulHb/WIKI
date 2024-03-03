import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from create_data import *


def visualize_image(file_path: str, cluster_num: int) -> None:
    """
    Display an image with a specified cluster number.

    Parameters:
    - file_path (str): Path to the image file.
    - cluster_num (int): Cluster number for visualization.

    Returns:
    - None
    """
    # Open the image file using the PIL library
    img = Image.open(file_path)

    # Display the image using matplotlib
    plt.imshow(img)

    # Set the title of the plot indicating the cluster number
    plt.title(f'Cluster {cluster_num} - Middle Image')

    # Turn off the axis for cleaner visualization
    plt.axis('off')

    # Show the plot
    plt.show()


# Function to get the middle value from a sorted DataFrame
def get_middle_value(sorted_df: pd.DataFrame) -> str:
    """
    Get the middle value from a sorted DataFrame.

    Parameters:
    - sorted_df (pd.DataFrame): DataFrame sorted by some criterion.

    Returns:
    - str: File path of the middle value.
    """
    # Calculate the index of the middle value
    middle_index = sorted_df.shape[0] // 2

    # Extract the file path of the middle value using the calculated index
    middle_value = sorted_df.iloc[middle_index]['FilePath']

    return middle_value


def pick_avereged_images(cluster_centers: np.ndarray, train_dataset: np.ndarray, df_data: pd.DataFrame, train_clusters: np.ndarray) -> Tuple[str, str, str, str]:
    """
    Select middle images from each cluster based on distance to cluster centers.

    Parameters:
    - cluster_centers (np.ndarray): Cluster centers obtained from KMeans.
    - train_dataset (np.ndarray): Training dataset.
    - df_data (pd.DataFrame): DataFrame containing file paths and cluster labels.
    - train_clusters (np.ndarray): Cluster labels assigned to training dataset.

    Returns:
    - Tuple of str: File paths of middle images for each cluster.
    """
    # Step 1: Calculate distances from each cluster center to each image vector
    distances = cdist(cluster_centers, train_dataset, metric='euclidean')

    # Step 2: Find the 5 closest images for each cluster center
    closest_images_ids = np.argsort(distances, axis=1)[:, :10]

    # Assuming df_val_clusters is your dataframe with 'FilePath' column
    # and closest_images_ids is an array of shape (num_clusters, 5) with indices of the closest images

    # Convert df_val_clusters to a dictionary for faster lookup
    file_paths_dict = df_data['FilePath'].to_dict()

    # Number of clusters and number of closest images
    num_clusters = closest_images_ids.shape[0]
    # Create a DataFrame to store FilePath and distances
    weighted_df = pd.DataFrame(columns=['FilePath'] + [f'Distance_to_Cluster_{i}' for i in range(num_clusters)])

    # Fill in the DataFrame
    for i in range(num_clusters):
        cluster_distances = distances[i, :]
        closest_image_indices = closest_images_ids[i, :]
        closest_image_paths = [file_paths_dict[idx] for idx in closest_image_indices]
        weighted_df[f'Distance_to_Cluster_{i}'] = cluster_distances
        weighted_df['FilePath'] = df_data.index
        weighted_df['Cluster'] = train_clusters
    # Display the resulting DataFrame
    cluster_0_distances = weighted_df[weighted_df["Cluster"] == 0][
        ['FilePath', 'Distance_to_Cluster_0', "Cluster"]].sort_values(by='Distance_to_Cluster_0')

    cluster_1_distances = weighted_df[weighted_df["Cluster"] == 1][
        ['FilePath', 'Distance_to_Cluster_1', "Cluster"]].sort_values(by='Distance_to_Cluster_1')

    cluster_2_distances = weighted_df[weighted_df["Cluster"] == 2][
        ['FilePath', 'Distance_to_Cluster_2', "Cluster"]].sort_values(by='Distance_to_Cluster_2')

    cluster_3_distances = weighted_df[weighted_df["Cluster"] == 3][
        ['FilePath', 'Distance_to_Cluster_3', "Cluster"]].sort_values(by='Distance_to_Cluster_3')
    # Get Middle values
    middle_value_cluster_0 = get_middle_value(cluster_0_distances)
    middle_value_cluster_1 = get_middle_value(cluster_1_distances)
    middle_value_cluster_2 = get_middle_value(cluster_2_distances)
    middle_value_cluster_3 = get_middle_value(cluster_3_distances)

    return middle_value_cluster_0, middle_value_cluster_1, middle_value_cluster_2, middle_value_cluster_3


def clusterization(data: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Perform KMeans clustering on the input data.

    Parameters:
    - data (np.ndarray): Input data for clustering.

    Returns:
    - Tuple of (pd.DataFrame, np.ndarray, np.ndarray):
        - df_result: DataFrame with file paths and cluster labels.
        - cluster_centers: Cluster centers obtained from KMeans.
        - train_clusters: Cluster labels assigned to the input data.
    """

    # Specify a range of clusters to test
    k_values = range(1, 11)

    # Fit KMeans for each value of k and calculate inertia
    inertia_values = []
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertia_values.append(kmeans.inertia_)
    # n_clusters is set to 4 based on the elbow method
    # Train KMeans on the training set with 4 clusters
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(data)

    # Predict clusters on the training set
    train_clusters = kmeans.predict(data)

    # Get cluster centers
    cluster_centers = kmeans.cluster_centers_
    df_result = pd.DataFrame({'FilePath': data.index, 'Cluster': train_clusters})
    return df_result, cluster_centers, train_clusters


def main(args: dict) -> None:
    """
    Main function to run the clustering and visualization process.

    Parameters:
    - args (dict): Command-line arguments.

    Returns:
    - None
    """
    # Retrieve command-line arguments or use default values
    input_path = args['input_path'] if args['input_path'] else INPUT_PATH
    confidence_th = args['confidence_th'] if args['confidence_th'] else CONFIDENCE_TH
    model_name = args['model_name'] if args['model_name'] else MODEL_NAME
    detector_backend = args['detector_backend'] if args['detector_backend'] else DETECTOR_BACKEND
    output_path = args['output_path'] if args['output_path'] else OUTPUT_PATH
    n_img = args['n_img'] if args['n_img'] else N_IMG

    # Define face size threshold
    face_size_th = (50, 50)

    # Save cropped images based on specified parameters
    save_cropped_images(path=input_path, output_path=output_path, confidence_th=confidence_th,
                        face_size_th=face_size_th, n_img=n_img)

    # Extract facial embeddings using specified parameters
    train_dataset, df_embeddings = make_embeddings(path=input_path, model_name=model_name,
                                                   detector_backend=detector_backend)

    # Perform KMeans clustering on the training dataset
    df_result, cluster_centers, train_clusters = clusterization(train_dataset)

    # Get the middle value (file path) for each cluster
    middle_value_cluster_0, middle_value_cluster_1, middle_value_cluster_2, middle_value_cluster_3 = pick_avereged_images(
        cluster_centers, train_dataset, df_result, train_clusters)

    # Visualize the middle images for each cluster
    visualize_image(middle_value_cluster_0, cluster_num=0)
    visualize_image(middle_value_cluster_1, cluster_num=1)
    visualize_image(middle_value_cluster_2, cluster_num=2)
    visualize_image(middle_value_cluster_3, cluster_num=3)


if __name__ == "__main__":
    # Command-line argument parser
    parser = argparse.ArgumentParser()

    # Add command-line arguments for specifying paths
    parser.add_argument("--input_path", type=str, help='Specify path for input data folder')
    # Add command-line arguments for specifying paths
    parser.add_argument("--confidence_th", type=float, help='Specify  confidence for model')
    # Add command-line arguments for specifying paths
    parser.add_argument("--model_name", type=str, help='Specify name of model')
    # Add command-line arguments for specifying paths
    parser.add_argument("--detector_backend", type=str, help='Specify name of detector backend')
    # Add command-line arguments for specifying paths
    parser.add_argument("--output_path", type=str, help='Specify path for output data folder')
    # Add command-line arguments for specifying paths
    parser.add_argument("--n_img", type=int, help='Specify number of images')
    # Parsing command-line arguments
    args = parser.parse_args()
    args = vars(args)

    # Calling the main function with parsed arguments
    main(args)