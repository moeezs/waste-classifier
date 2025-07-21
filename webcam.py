#!/usr/bin/env python3
"""
Real-time Waste Classification using Webcam

This script uses a trained model to classify waste in real-time from webcam feed.
It overlays text indicating whether the detected waste is "Recyclable" or "Organic".
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

class WasteClassifier:
    def __init__(self, model_path='best_waste_classifier.h5'):
        """Initialize the waste classifier"""
        self.model_path = model_path
        self.model = None
        self.img_size = (224, 224)
        self.class_names = ['Organic', 'Recyclable']  # 0: Organic, 1: Recyclable
        self.confidence_threshold = 0.7
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            self.model = load_model(self.model_path)
            print("Model loaded successfully!")
        else:
            print(f"Error: Model file {self.model_path} not found!")
            print("Please run main.py first to train the model.")
            return False
        return True
    
    def preprocess_image(self, image):
        """Preprocess image for prediction"""
        # Resize image
        image = cv2.resize(image, self.img_size)
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Normalize pixel values
        image = image.astype('float32') / 255.0
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        return image
    
    def predict(self, image):
        """Make prediction on image"""
        if self.model is None:
            return None, 0.0
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Make prediction
        prediction = self.model.predict(processed_image, verbose=0)[0][0]
        
        # Convert to class and confidence
        if prediction > 0.5:
            class_name = self.class_names[1]  # Recyclable
            confidence = prediction
        else:
            class_name = self.class_names[0]  # Organic
            confidence = 1 - prediction
        
        return class_name, confidence
    
    def draw_prediction(self, frame, class_name, confidence):
        """Draw prediction text on frame"""
        # Prepare text
        text = f"{class_name}: {confidence:.2f}"
        
        # Set colors based on class
        if class_name == "Recyclable":
            color = (0, 255, 0)  # Green for recyclable
            bg_color = (0, 100, 0)
        else:
            color = (0, 165, 255)  # Orange for organic
            bg_color = (0, 82, 127)
        
        # Text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        
        # Get text dimensions
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(frame, 
                     (10, 10), 
                     (text_width + 20, text_height + baseline + 20), 
                     bg_color, -1)
        
        # Draw text
        cv2.putText(frame, text, (15, text_height + 15), 
                   font, font_scale, color, thickness)
        
        # Add confidence indicator
        if confidence > self.confidence_threshold:
            status = "HIGH CONFIDENCE"
            status_color = (0, 255, 0)
        else:
            status = "LOW CONFIDENCE"
            status_color = (0, 255, 255)
        
        cv2.putText(frame, status, (15, text_height + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        return frame
    
    def run_webcam(self, camera_id=0):
        """Run real-time classification from webcam"""
        if self.model is None:
            print("Model not loaded. Cannot start webcam.")
            return
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("Starting webcam classification...")
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press 's' to save current frame")
        print("- Press 'c' to change camera (if multiple cameras available)")
        
        frame_count = 0
        
        while True:
            # Capture frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Make prediction every few frames to reduce computational load
            if frame_count % 3 == 0:  # Predict every 3rd frame
                class_name, confidence = self.predict(frame)
            
            # Draw prediction on frame
            if 'class_name' in locals():
                frame = self.draw_prediction(frame, class_name, confidence)
            
            # Add instructions
            cv2.putText(frame, "Press 'q' to quit", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Waste Classifier', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"captured_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")
            elif key == ord('c'):
                # Try to switch camera
                cap.release()
                camera_id = (camera_id + 1) % 2  # Toggle between 0 and 1
                cap = cv2.VideoCapture(camera_id)
                print(f"Switched to camera {camera_id}")
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam classification stopped.")

def main():
    """Main function"""
    # Check for available model files
    model_files = ['best_waste_classifier.h5', 'waste_classifier_final.h5']
    model_path = None
    
    for model_file in model_files:
        if os.path.exists(model_file):
            model_path = model_file
            break
    
    if model_path is None:
        print("No trained model found!")
        print("Available options:")
        print("1. Run main.py first to train a new model")
        print("2. Copy a trained model file to this directory")
        return
    
    # Create classifier
    classifier = WasteClassifier(model_path)
    
    # Start webcam classification
    classifier.run_webcam()

if __name__ == "__main__":
    main()
