# MultiModal_Cursor_Controller
Multimodal Maestro offers a new method of human computer interaction by fusing two input modalities-gaze interaction and hand gesture recognition.

# Our Published Research Paper 
Research Paper: https://ieeexplore.ieee.org/document/10969887

# Multimodal Maestro

## Introduction
Multimodal Maestro is an advanced eye-gaze and hand gesture control system that eliminates the need for physical input devices. The project integrates computer vision and deep learning techniques to provide a smooth, accessible, and traditional input device-free human-computer interaction experience.

## Objectives

### Objective 1: Eye-Gaze Cursor Control
- Implements an advanced facial landmark detection algorithm to track and analyze eye-gaze fixations.
- Maps gaze coordinates onto the computer screen for fine-grained cursor control.
- Smoothness algorithms ensure jerk-free cursor navigation.
- Uses Eye Aspect Ratio (EAR) for blink detection, mapping left and right blinks to left and right clicks.
- Utilizes VGG-19 for spatial feature extraction and LSTM for sequence processing to classify eye states.
- Model augmentation improves performance across real-world variations, making it inclusive for users with limited mobility.

### Objective 2: Hand Gesture Control
- Uses OpenCV and machine learning models (CNN, LSTM, MobileNetV2, VGG-19) for gesture recognition.
- Tracks hand movements using a webcam and maps specific gestures to cursor actions like left-click, right-click, and scrolling.
- Designed to be adaptive and responsive based on real-time user feedback.
- Ensures accessibility with an engaging and seamless user experience.

### Objective 3: Multimodal Integration
A combined approach using both gaze and hand gesture recognition:

- Integrated Interface:
  - Built with HTML, CSS, and JavaScript.
  - Allows users to choose between Hand Gesture and Gaze Controller.
  - Displays a slideshow explaining module functions.
  - Includes an "About" tab and a "Team Members" section.

- Backend Connectivity:
  - Uses Flask to handle real-time interactivity.
  - Provides error handling through alerts for a smooth user experience.

## Technologies Used
- Frontend: HTML, CSS, JavaScript
- Backend: Flask
- Computer Vision: OpenCV, Mediapipe
- Machine Learning: CNN, LSTM, MobileNetV2, VGG-19
- Libraries: NumPy, TensorFlow/Keras, PyTorch

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/multimodal-maestro.git
   cd multimodal-maestro
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Flask server:
   ```sh
   python app.py
   ```
4. After running app.py terminal will display two links click on the first link, it will redirect you to browser where you can interact with buttons to control the mouse/cursor.

## Usage
- Choose between Hand Gesture or Gaze Controller from the interface.
- Ensure proper lighting for better performance.
- Ensure the webcam is properly positioned for accurate tracking.

## Contributing
Pull requests are welcome. For significant changes, open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Team Members: Akanksha S, Sandeep Telkar R.
- Special thanks to open-source contributors and research papers that inspired this project.
