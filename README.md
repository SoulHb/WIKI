## WIKI face clusterization

### Overview
This documentation provides information about the method of clusterization faces using face detectors and unsupervised learning algorithm. It also includes usage instructions and author information.


### Data
The Dataset provides random images of people and objects.
[Link to the dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/)
## Model Architecture
We used a Facenet model with opencv detector backed as face extractor and embedding model, then we pass extracted faces though PCA and train Kmeans with k=4 on processed data.

## Usage
### Requirements
- Ubuntu 20.04
- Python 3.10

### Getting Started
Clone repository
```bash
git clone https://github.com/SoulHb/WIKI.git
```
Move to project folder
```bash
cd WIKI
```
Create conda env 
```bash
conda create --name=wiki python=3.10
```
Activate virtual environment
```bash
conda activate wiki
```
Install pip 
```bash
conda install pip 
```
Install dependencies
```bash
pip install -r requirements.txt
```

## Inference
To use the  WIKI face clusterization, follow the instructions below:

Move to src folder
```bash
cd src
```
Run inference
```bash
python inference.py --input_path INPUT_PATH --confidence_th CONFIDENCE_TH --model_name MODEL_NAME --detector_backend DETECTOR_BACKEND --output_path OUTPUT_PATH --n_img N_IMG
```

## Author
This WIKI face clusterization project was developed by 'Полум*я' team. If you have any questions, please contact teamleader: namchuk.maksym@gmail.com
