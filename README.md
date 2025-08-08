# 🎮 Rock Paper Scissors AI Emotion Detector

An interactive Rock Paper Scissors game that uses computer vision and AI to detect hand gestures and facial emotions in real-time. The game features adaptive UI scaling, multiple camera support, and emotion-based reactions.

![Python](https://img.shields.io/badge/python-v3.7+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.7-orange.svg)

## ✨ Features

### 🎯 Core Gameplay
- **Real-time Hand Gesture Recognition**: Detects Rock, Paper, Scissors gestures using MediaPipe
- **Facial Emotion Detection**: Recognizes 5 emotions (Happy, Sad, Surprised, Sleepy, Neutral)
- **Emotion-Based Reactions**: Game responses change based on your detected emotion
- **Score Tracking**: Keeps track of wins, losses, and draws

### 📷 Camera & Display
- **Multi-Camera Support**: Automatic detection and selection of available cameras
- **Adaptive Resolution**: Tests and selects the best available camera resolution
- **Auto-Scaling UI**: Interface automatically adjusts based on camera resolution
- **Interactive Camera Selection**: Terminal-based camera chooser with resolution display

### 🎨 Visual Features
- **Adaptive Text Scaling**: Text size adjusts for different screen resolutions
- **Toggle Landmarks**: Press 'L' to show/hide facial and hand landmarks
- **Countdown Timer**: 3-second countdown with pulsing animation
- **Professional UI**: Semi-transparent overlays with color-coded information

### 🔧 Technical Features
- **Cross-Platform Audio**: Beep sounds on Windows and Linux/Mac
- **Camera Warm-up**: Ensures stable camera operation
- **Error Handling**: Robust camera detection and fallback mechanisms
- **Memory Efficient**: Optimized for real-time performance

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- Webcam or external camera
- Windows, macOS, or Linux

### Step 1: Clone or Download
```bash
git clone <repository-url>
cd Object\ Detection\ with\ python
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Game
```bash
python live_rsp.py
```

## 🎮 How to Play

### Starting the Game
1. **Launch**: Run `python live_rsp.py`
2. **Select Camera**: Choose from available cameras in the terminal menu
3. **Position Yourself**: Make sure your face is visible to the camera
4. **Start Round**: The game automatically starts when it detects your face

### Gameplay
1. **Countdown**: A 3-second countdown will begin
2. **Make Gesture**: Show your hand gesture (Rock, Paper, or Scissors) after countdown
3. **Emotion Detection**: Your facial expression will be analyzed
4. **Results**: See who won with emotion-based commentary
5. **Continue**: Press 'R' to play again or 'Q' to quit

### Controls
| Key | Action |
|-----|--------|
| `R` | Start new round |
| `Q` | Quit game |
| `L` | Toggle landmarks display |

## 🎭 Emotion Detection

The AI can detect and respond to these emotions:

### 😊 Happy
- **Detection**: Upward mouth curve (smile)
- **Reactions**: Encouraging and positive responses

### 😢 Sad  
- **Detection**: Downward mouth curve (frown)
- **Reactions**: Comforting and supportive messages

### 😮 Surprised
- **Detection**: Wide eyes + raised eyebrows + open mouth + dropped lower lip
- **Reactions**: Excited and surprised commentary

### 😴 Sleepy
- **Detection**: Nearly closed eyes (low eye aspect ratio)
- **Reactions**: Playful tired-themed responses

### 😐 Neutral
- **Detection**: Default state when no strong emotion detected
- **Reactions**: Standard game responses

## 🤲 Hand Gestures

The game recognizes these hand gestures:

| Gesture | Description | Hand Position |
|---------|-------------|---------------|
| ✊ **Rock** | Closed fist | All fingers down |
| ✋ **Paper** | Open palm | All fingers extended |
| ✌️ **Scissors** | Peace sign | Index and middle finger up |

## 🖥️ System Requirements

### Minimum Requirements
- **OS**: Windows 7/10/11, macOS 10.12+, Ubuntu 16.04+
- **RAM**: 4GB
- **Camera**: Any USB webcam or built-in camera
- **Python**: 3.7+

### Recommended Requirements  
- **OS**: Windows 10/11, macOS 12+, Ubuntu 20.04+
- **RAM**: 8GB or more
- **Camera**: HD webcam (720p or higher)
- **Python**: 3.8+

## 🛠️ Technical Details

### Dependencies
- **OpenCV**: Computer vision library for camera handling and image processing
- **MediaPipe**: Google's framework for hand and face landmark detection
- **NumPy**: Numerical computing (automatically installed with OpenCV)

### Camera Support
- **Resolution Testing**: Automatically tests resolutions from 1920x1080 down to 320x240
- **Multi-Camera**: Supports up to 5 cameras (indices 0-4)
- **Fallback**: Graceful degradation if preferred resolution isn't available

### Performance
- **FPS**: Typically 15-30 FPS depending on camera and system
- **Latency**: Near real-time gesture and emotion detection
- **Memory**: Approximately 200-400MB RAM usage

## 🐛 Troubleshooting

### Camera Issues
**Problem**: "No cameras found!"
- **Solution**: Check camera connections and permissions
- **Windows**: Ensure camera isn't being used by another application
- **macOS**: Grant camera permissions in System Preferences
- **Linux**: Install v4l-utils: `sudo apt install v4l-utils`

### Performance Issues
**Problem**: Low FPS or laggy detection
- **Solution**: 
  - Close other camera applications
  - Use a lower resolution camera
  - Reduce background applications

### Detection Issues
**Problem**: Gestures or emotions not detected
- **Solution**:
  - Ensure good lighting
  - Position hands clearly in camera view
  - Make distinct facial expressions
  - Toggle landmarks (press 'L') to see detection points

## 📝 Development

### File Structure
```
Object Detection with python/
├── live_rsp.py           # Main game file
├── requirements.txt      # Python dependencies
├── README.md            # This file
└── hand_landmarker.task # MediaPipe model file
```

### Key Functions
- `detect_emotion()`: Analyzes facial landmarks for emotion detection
- `get_hand_gesture()`: Recognizes hand gestures from landmarks
- `choose_camera()`: Interactive camera selection interface
- `find_best_camera_resolution()`: Automatic resolution optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

## 🙏 Acknowledgments

- **MediaPipe**: Google's MediaPipe framework for landmark detection
- **OpenCV**: Open Source Computer Vision Library
- **Community**: Thanks to all contributors and testers

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are correctly installed
3. Verify camera permissions and availability
4. Test with different lighting conditions

---

**Enjoy playing Rock Paper Scissors with AI! 🎮✨**
