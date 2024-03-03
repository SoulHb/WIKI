# List of supported face recognition models
models = [
    "VGG-Face",
    "Facenet",
    "Facenet512",
    "OpenFace",
    "DeepFace",
    "DeepID",
    "ArcFace",
    "Dlib",
    "SFace",
]

# List of supported face detection backends
backends = [
    'opencv',
    'ssd',
    'dlib',
    'mtcnn',
    'retinaface',
    'mediapipe',
    'yolov8',
    'yunet',
    'fastmtcnn',
]

# General data folder path
INPUT_PATH = './wiki_crop'

# Default face recognition model
MODEL_NAME = 'Facenet'

# Default number of images to extract faces from
N_IMG = 1

# Default face detection backend
DETECTOR_BACKEND = 'opencv'

# Default confidence threshold for face extraction
CONFIDENCE_TH = 0.95

# Default output path for saving extracted faces
OUTPUT_PATH = 'extracted_faces'
