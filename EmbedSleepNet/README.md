# EmbedSleepNet
This repository reproduces TinySleepNet for sleep stage prediction based on the signal channel EEG
using PyTorch and implements a new smaller and faster network called EmbedSleepNet.
# Environment Setup
1. Create a virtual environment with Python 3.8:
`virtualenv venv --python=python3.8`
2. Activate the environment:
`source venv/bin/activate` or `venv\Scripts\activate.bat` (for Windows)
3. Install dependencies:
`pip install -r requirements.txt`
4. Download and extract Sleep-EDF dataset from https://www.physionet.org/content/sleep-edfx/1.0.0/
5. Preprocess the data by running `python preprocess.py --data_path PATH_TO_sleep-cassete_FOLDER`
# Running
To start the training process simply run:
`python train.py --flavor=[embed, tiny]`, where tiny represents original TinySleepNet, 
and embed represents newly introduced EmbedSleepNet. 
You may additionally specify the number of epochs and model output name, for example:

`python train.py --flavor embed --epochs 450 --model-name model`