# Action Detection using MediaPipe and LSTM

A real-time action detection system that uses MediaPipe for pose landmark extraction and LSTM neural networks to classify human actions from webcam feed.

## Project Overview

This project implements an action recognition system capable of detecting 4 different actions:
- **hello**: Greeting gesture
- **thanks**: Thank you gesture  
- **iloveyou**: I love you gesture
- **Feed**: Feeding gesture

The system uses MediaPipe Holistic to extract 3D pose, face, and hand landmarks from video frames, then feeds these features into an LSTM model for sequence-based action classification.

## Features

- Real-time webcam-based action detection
- MediaPipe Holistic for robust landmark extraction (pose, face, hands)
- LSTM deep learning model for temporal sequence recognition
- Visual feedback with styled landmarks and probability display
- Frame sequence collection for training data

## Technologies Used

- **Python 3.7+**
- **OpenCV**: Video capture and display
- **MediaPipe**: Pose and hand landmark detection
- **TensorFlow/Keras**: LSTM model building and training
- **NumPy**: Numerical operations
- **Scikit-learn**: Data preprocessing and metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/the-robotron/testingml1.git
cd testingml1

# Install required packages
pip install opencv-python mediapipe tensorflow numpy scikit-learn matplotlib
```

## Project Structure

```
testingml1/
│
├── Action Detection Tutorial.ipynb    # Main notebook with full pipeline
├── MP_Data/                          # Training data directory (sequences)
├── Logs/                             # Model training logs
└── README.md                         # This file
```

## Usage

### Training the Model

1. Open `Action Detection Tutorial.ipynb` in Jupyter Notebook
2. Run data collection cells to record action sequences (30 sequences × 30 frames per action)
3. Train the LSTM model on collected sequences
4. Model will be saved with training logs

### Real-time Detection

Run the detection cells in the notebook to:
1. Initialize webcam and MediaPipe
2. Load trained LSTM model  
3. Perform real-time action classification
4. Display results with visual feedback

## Model Architecture

```
LSTM(64) → LSTM(128) → LSTM(64) → Dense(64) → Dense(32) → Dense(4)
```

- **Input**: Sequence of 30 frames with 1662 features per frame
- **Output**: 4-class probability distribution
- **Loss**: Categorical crossentropy
- **Optimizer**: Adam

## Known Issues and Fixes Applied

### Issue 1: IndexError in prob_viz() Function
**Problem**: Colors list had only 3 colors but 4 actions, causing IndexError  
**Status**: Identified - needs color added  
**Fix**: Add 4th color to colors list

### Issue 2: Model Performance
**Problem**: Training accuracy stuck at ~38.6%, indicating poor learning  
**Status**: Identified - architecture needs optimization  
**Potential Solutions**:
- Increase training data (more sequences per action)
- Add dropout layers to prevent overfitting
- Experiment with different LSTM units
- Adjust learning rate
- Normalize input features

### Issue 3: Missing Video Paths
**Problem**: Video file paths (0.npy, 1.npy, etc.) referenced but not in repo  
**Status**: Non-critical - training data format

## Data Collection

The system collects 30 video sequences per action, with each sequence containing 30 frames:
- Total sequences: 120 (4 actions × 30 sequences)
- Frames per sequence: 30
- Features per frame: 1662 (pose + face + hand landmarks)

## Future Improvements

- [ ] Fix color list bug in visualization
- [ ] Improve model accuracy through architecture optimization
- [ ] Add more action classes
- [ ] Implement data augmentation
- [ ] Create standalone deployment script
- [ ] Add model checkpointing and early stopping

## Requirements

```
opencv-python>=4.5.0
mediapipe>=0.8.9
tensorflow>=2.8.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
```

## Acknowledgments

- MediaPipe by Google for pose estimation
- TensorFlow team for deep learning framework

## License

MIT License

## Author

the-robotron
