import os
import torch.nn as nn
import cv2
import re
import numpy as np
import pandas as pd
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
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


def save_cropped_images(path: str, output_path: str, n_img: int, cascade: str) -> None:
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
    # Check if the output path exists; if not, create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Get a list of all folders in the specified path
    list_folders = os.listdir(path)
    images_list = []  # List to store all image paths
    new_images_list = []  # List to store selected images

    # Iterate through each folder and collect image paths
    for folder in list_folders:
        subfolder_path = os.path.join(path, folder)
        temp_list = os.listdir(subfolder_path)
        for img_path in temp_list:
            images_list.append(os.path.join(subfolder_path, img_path))

    # Process each image path
    for path_im in images_list:
        # Check if the desired number of images is reached
        if len(new_images_list) == n_img:
            break

        # Open the image and convert it to RGB format
        frame = Image.open(path_im).convert('RGB')

        # Check if the image has height equal to 1 (potential issue)
        if (frame.height == 1):
            continue

        # Convert the image to grayscale
        gray = np.array(frame)
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

        # Load the Haar cascade classifier for frontal faces
        face_frontal_detector = cv2.CascadeClassifier(cv2.data.haarcascades + cascade)

        # Detect faces in the grayscale image
        faces = face_frontal_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # If faces are detected, save the image to the output path
        if len(faces) != 0:
            new_images_list.append(os.path.join(output_path, path_im))

            # Define a regular expression pattern to extract the filename
            pattern = r'.*/(\d+_\d{4}-\d{2}-\d{2}_\d{4}\.jpg)'

            # Use re.search to find the match
            match = re.search(pattern, os.path.join(output_path, path_im))

            # Check if there's a match and extract the filename
            if match:
                filename = match.group(1)
            else:
                print("No match found.")

            # Save the image with the extracted filename to the output path
            frame.save(os.path.join(output_path, filename), "JPEG")


class Embeddings(nn.Module):
    def __init__(self, mtcnn: nn.Module, embedding_model: nn.Module):
        super(Embeddings, self).__init__()
        self.mtcnn = mtcnn
        self.embedding_model = embedding_model
        self.embedding_model.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply MTCNN to get cropped face image
        img_cropped = self.mtcnn(x)

        # Check if a face is detected
        if img_cropped is not None:
            # Apply the embedding model to get the facial embeddings
            img_embedding = self.embedding_model(img_cropped.unsqueeze(0).to(DEVICE))

            # Convert the tensor to a numpy array and return
            return img_embedding.squeeze(0).detach().cpu().numpy()


def make_embeddings(path: str, pipeline: Embeddings) -> pd.DataFrame:
    # Get a list of image paths in the specified directory
    images_list = os.listdir(path)

    # Lists to store extracted features and corresponding image names
    feature_list = []
    new_images_list = []

    # Iterate through each image path
    for img_path in images_list:
        # Open the image using PIL
        frame = Image.open(os.path.join(path, img_path))

        # Get facial embeddings using the provided pipeline
        embedding = pipeline(frame)

        # Check if embeddings are obtained
        if embedding is not None:
            # Append the embeddings and corresponding image name to the lists
            feature_list.append(embedding)
            new_images_list.append(img_path)

    # Create a DataFrame with the extracted features and image names
    df_data = pd.DataFrame(feature_list, columns=range(512), index=new_images_list)

    # Return the DataFrame
    return df_data