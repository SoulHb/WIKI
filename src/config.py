import torch

# General data folder path
INPUT_PATH = './wiki_crop'

# Default face recognition model
MODEL_NAME = 'vggface2'

# Default number of images to extract faces from
N_IMG = 50

# Default face detection backend
DETECTOR_BACKEND = 'haarcascade_frontalface_default.xml'

# Default output path for saving extracted faces
OUTPUT_PATH = './extracted_faces'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
