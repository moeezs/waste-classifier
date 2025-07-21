# Waste Classification System

A machine learning system that classifies waste into **Organic** and **Recyclable** categories using computer vision.

## 🗂️ Project Structure

```
waste-classifier/
├── main.py              # Model training script
├── webcam.py            # Real-time webcam classification
├── deploy.sh            # SSH deployment script
├── requirements.txt     # Python dependencies
├── DATASET/            # Training dataset
│   └── DATASET/
│       ├── TRAIN/
│       │   ├── O/      # Organic waste images
│       │   └── R/      # Recyclable waste images
│       └── TEST/
│           ├── O/      # Organic waste test images
│           └── R/      # Recyclable waste test images
└── README.md           # This file
```

## 🚀 Quick Start

### 1. Train the Model

```bash
python main.py
```

This will:
- Load and preprocess the dataset
- Create a CNN model
- Train for up to 50 epochs with early stopping
- Save the best model as `best_waste_classifier.h5`
- Generate training plots

### 2. Run Webcam Classification

```bash
python webcam.py
```

This will:
- Load the trained model
- Start webcam feed
- Display real-time predictions with confidence scores
- Show "Recyclable" (green) or "Organic" (orange) labels

### Controls:
- Press `q` to quit
- Press `s` to save current frame
- Press `c` to switch camera (if multiple available)

## 📦 Installation

### Local Setup

1. **Clone/Download the project**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🤖 Model Details

- **Architecture:** Convolutional Neural Network (CNN)
- **Input:** 224x224 RGB images
- **Output:** Binary classification (Organic vs Recyclable)
- **Framework:** TensorFlow/Keras
- **Data Augmentation:** Rotation, shifts, zoom, flip

### Model Layers:
1. 4 Convolutional blocks with BatchNormalization
2. MaxPooling layers for downsampling
3. Fully connected layers with Dropout
4. Sigmoid output for binary classification

## 📊 Dataset

- **Source:** Kaggle Waste Classification Dataset
- **Classes:** 
  - `O` - Organic waste (food scraps, biodegradable materials)
  - `R` - Recyclable waste (plastic, metal, glass, paper)
- **Training Images:** ~20,000 images
- **Test Images:** ~5,000 images

## 🔧 Configuration

### Model Parameters (main.py):
```python
IMG_SIZE = (224, 224)    # Input image size
BATCH_SIZE = 32          # Training batch size
EPOCHS = 30              # Maximum training epochs
LEARNING_RATE = 0.001    # Adam optimizer learning rate
```

### Webcam Parameters (webcam.py):
```python
confidence_threshold = 0.7  # Confidence threshold for "high confidence"
prediction_frequency = 3    # Predict every N frames (for performance)
```

## 📱 Usage Examples

### Training on Custom Dataset
1. Replace the `DATASET` folder with your own images
2. Organize as: `TRAIN/CLASS_NAME/` and `TEST/CLASS_NAME/`
3. Update class names in `webcam.py` if needed
4. Run `python main.py`

### Using External Camera
The webcam script automatically detects available cameras. Press `c` to cycle through them.

### Running on Raspberry Pi
1. Use the deployment script to transfer files
2. Enable camera module: `sudo raspi-config`
3. Install additional dependencies if needed:
   ```bash
   sudo apt install python3-picamera
   ```

## 🎯 Performance Tips

1. **GPU Acceleration:** Install `tensorflow-gpu` for faster training
2. **Memory:** Reduce batch size if getting OOM errors
3. **Speed:** Lower prediction frequency in webcam mode
4. **Accuracy:** Ensure good lighting and clear view of objects

## 🔍 Troubleshooting

### Common Issues:

**Model not found:**
- Run `main.py` first to train the model
- Check that `.h5` files are in the same directory

**Camera not working:**
- Try different camera IDs (0, 1, 2...)
- Check camera permissions
- For SSH: use X11 forwarding (`ssh -X`)

**Import errors:**
- Install requirements: `pip install -r requirements.txt`
- Use virtual environment for clean installation

**Low accuracy:**
- Train for more epochs
- Increase dataset size
- Improve image quality/lighting

## 📝 Files Generated

After training:
- `best_waste_classifier.h5` - Best model during training
- `waste_classifier_final.h5` - Final trained model
- `training_history.png` - Training plots

During webcam use:
- `captured_frame_*.jpg` - Saved frames (press 's')

## 🌟 Features

- ✅ Real-time classification
- ✅ Confidence scoring
- ✅ Visual feedback with color coding
- ✅ Easy deployment to remote servers
- ✅ Camera switching capability
- ✅ Frame capture functionality
- ✅ Comprehensive training monitoring

## 🔄 Future Improvements

- [ ] Add more waste categories (glass, metal, paper, etc.)
- [ ] Implement object detection for multiple items
- [ ] Add voice feedback
- [ ] Create mobile app version
- [ ] Add recycling statistics tracking

## 📄 License

This project is open source. Feel free to modify and distribute.

---

**Happy Waste Sorting! ♻️🌱**
