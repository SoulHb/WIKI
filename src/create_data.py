import os
import cv2
import pandas as pd
from deepface import DeepFace
from sklearn.decomposition import PCA
from config import *
from typing import Tuple


def feature_extractor(df_data: pd.DataFrame) -> pd.DataFrame:
    """
    Perform PCA on the input DataFrame to reduce the number of features.

    Parameters:
    - df_data (pd.DataFrame): Input DataFrame containing features.

    Returns:
    - pd.DataFrame: DataFrame with reduced features after PCA.
    """
    # Initialize PCA with the target explained variance of 99%
    pca = PCA(n_components=0.99)

    # Fit the PCA model to the input data
    pca.fit(df_data)

    # Transform the data to the reduced space based on the fitted PCA model
    df_data_reduced = pca.transform(df_data)

    return df_data_reduced


def save_cropped_images(path: str, output_path: str, confidence_th: float, face_size_th: Tuple[int, int], n_img: int) -> None:
    """
    Save cropped faces from input images to the specified output path.

    Parameters:
    - path (str): Input path containing images.
    - output_path (str): Output path to save cropped faces.
    - confidence_th (float): Confidence threshold for face extraction.
    - face_size_th (Tuple[int, int]): Minimum face size (height, width).
    - n_img (int): Maximum number of images to process.

    Returns:
    - None
    """
    file_list = []
    counter = 1

    # Check if the output path exists; if not, create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Traverse the directory structure to get a list of all files in the specified path
    for root, dirs, files in os.walk(path):
        file_list.extend([os.path.join(root, file) for file in files])

    # Iterate through each file in the list
    for file in file_list:
        try:
            # Extract faces from the image using DeepFace
            face = DeepFace.extract_faces(img_path=file, target_size=(224, 224),
                                          detector_backend=backends[4])  # [0]['face']

            # Check if the extracted face meets specified confidence and size criteria
            if face[0]['confidence'] >= confidence_th and face[0]['facial_area']['h'] >= face_size_th[0] and \
                    face[0]['facial_area']['w'] >= face_size_th[1]:

                # Convert the face to RGB format and scale values
                face = cv2.cvtColor(face[0]['face'], cv2.COLOR_BGR2RGB) * 255.0

                # Extract the file name and save the face image to the output path
                file_name = file.split('\\')[-1]
                cv2.imwrite(os.path.join(output_path, file_name), face)

                # Update the counter and break if the desired number of images is reached
                if counter < n_img:
                    counter += 1
                else:
                    break
        except Exception as e:
            # Ignore exceptions and continue processing other images
            pass


def make_embeddings(path: str, model_name: str, detector_backend: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract facial embeddings from images in the specified path.

    Parameters:
    - path (str): Input path containing images.
    - model_name (str): Name of the face recognition model.
    - detector_backend (str): Backend for face detection.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]: Tuple containing two DataFrames -
        1. DataFrame with reduced features after PCA.
        2. DataFrame with raw embeddings for each image.
    """
    # Create an empty DataFrame with 128 columns to store embeddings
    df_data = pd.DataFrame(columns=range(128))

    # Iterate through files in the specified path
    for file in os.listdir(path):
        file_path = os.path.join(path, file)  # Get the full file path

        try:
            # Extract embeddings from the image file using DeepFace
            embedding = DeepFace.represent(img_path=file_path, model_name=model_name, detector_backend=detector_backend)

            # Add the extracted embedding to the DataFrame, using the file name as index
            df_data.loc[file] = embedding[0]['embedding']

            # Print progress every 250 images
            if len(df_data) % 250 == 0:
                print(len(df_data), "images processed")

        except Exception as e:
            # Ignore exceptions and continue processing other images
            pass
    # Reduced dataset
    df_reduced = feature_extractor(df_data)
    return df_reduced, df_data
