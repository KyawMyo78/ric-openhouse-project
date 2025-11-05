# 🎮 Hand Gesture Games Collection

An interactive game collection featuring Rock Paper Scissors, Virtual Air Drawing, and Fruit Ninja - all controlled with hand gestures using computer vision and AI. Play games, draw in the air, and slice fruits with your hands!

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.1-green.svg)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.7-orange.svg)
![Pygame](https://img.shields.io/badge/Pygame-2.5.0-yellow.svg)

## ✨ Features

### 🕹️ Game Launcher (openhouse.py)
- **Unified Interface**: Launch all games from one convenient menu
- **Multi-Camera Support**: Choose your preferred camera before starting any game
- **Quick Navigation**: Back buttons to return to menu
- **Fast Launch**: Camera selection remembered during session

### 🎯 Rock Paper Scissors (live_rsp.py)
- **Real-time Hand Gesture Recognition**: Detects Rock, Paper, Scissors gestures using MediaPipe
- **Facial Emotion Detection**: Recognizes 5 emotions (Happy, Sad, Surprised, Sleepy, Neutral)
- **Emotion-Based Reactions**: Game responses change based on your detected emotion
- **Score Tracking**: Keeps track of wins, losses, and draws
- **Toggle Landmarks**: Press 'L' to show/hide facial and hand landmarks

### 🎨 Virtual Air Drawing (draw.py)
- **Draw in Mid-Air**: Use your index finger to draw on a virtual canvas
- **Shape Tools**: Circle, Rectangle, Triangle, and Freehand drawing
- **Color Palette**: Multiple colors to choose from
- **Two-Hand Gestures**:
  - Both hands pinching = Resize objects
  - Right pinch + Left L-shape = Rotate objects
- **Select & Move**: Pinch gesture to select and move drawn objects
- **Hand Restriction**: Right hand only for drawing, left hand assists with transformations
- **Undo Feature**: Undo last drawing action

### 🍉 Fruit Ninja (fruit_ninja.py)
- **Slice Fruits**: Swipe your hand to slice falling fruits
- **Avoid Bombs**: Don't hit the bombs or lose lives
- **Combo System**: Build combos for bonus points
- **Score Tracking**: Track your high score
- **Sound Effects**: Satisfying slice sounds and combos
- **Lives System**: 3 lives to keep playing

### 📷 Camera & Display
- **Multi-Camera Support**: Automatic detection and selection of available cameras
- **Adaptive Resolution**: Tests and selects the best available camera resolution
- **Auto-Scaling UI**: Interface automatically adjusts based on camera resolution
- **Camera Mirroring**: Natural mirrored view for intuitive interaction

### 🔧 Technical Features
- **Two-Hand Tracking**: Supports simultaneous tracking of both hands
- **Gesture Recognition**: Advanced gesture detection (pinch, L-shape, palm, etc.)
- **Real-time Processing**: Optimized for smooth performance
- **Cross-Platform Audio**: Sound effects on all platforms
- **Error Handling**: Robust camera detection and fallback mechanisms

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam or external camera
- Windows, macOS, or Linux

### Step 1: Clone or Download
```bash
git clone https://github.com/KyawMyo78/rock-paper-scissor-with-python.git
cd "Object Detection with python"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run the Game Launcher
```bash
python openhouse.py
```

**Or run individual games:**
```bash
python live_rsp.py      # Rock Paper Scissors
python draw.py          # Virtual Air Drawing
python fruit_ninja.py   # Fruit Ninja
```

## 🎮 How to Play

### 🕹️ Game Launcher
1. **Launch**: Run `python openhouse.py`
2. **Select Camera**: Choose from available cameras
3. **Choose Game**: Select Rock Paper Scissors, Virtual Air Drawing, or Fruit Ninja
4. **Play**: Enjoy the game!
5. **Back Button**: Return to menu anytime

### 🎯 Rock Paper Scissors
1. **Position Yourself**: Make sure your face is visible
2. **Countdown**: Wait for 3-second countdown
3. **Make Gesture**: Show Rock, Paper, or Scissors
4. **Results**: See who won with emotion-based commentary
5. **Controls**: `R` = New round, `Q` = Quit, `L` = Toggle landmarks

### 🎨 Virtual Air Drawing
1. **Right Hand Controls**:
   - **1 Finger**: Draw with index finger
   - **Pinch**: Select and move objects
   - **3 Fingers**: Show menu
   - **Palm**: Stop action
2. **Two Hands Controls**:
   - **Both Pinching**: Resize selected object (spread/squeeze hands)
   - **Right Pinch + Left L-shape**: Rotate selected object
3. **UI Selection**: Point at colors/shapes to select
4. **Keyboard**: `C` = Clear, `H` = Help, `U` = Undo, `Q` = Quit

### 🍉 Fruit Ninja
1. **Swipe**: Move your hand to slice fruits
2. **Avoid Bombs**: Don't hit red bombs
3. **Build Combos**: Slice multiple fruits quickly
4. **Lives**: You have 3 lives
5. **Controls**: `R` = Restart, `ESC` = Quit

### Controls Summary
| Game | Key | Action |
|------|-----|--------|
| RPS | `R` | New round |
| RPS | `Q` | Quit |
| RPS | `L` | Toggle landmarks |
| Draw | `C` | Clear canvas |
| Draw | `H` | Toggle help |
| Draw | `U` | Undo |
| Draw | `Q` | Quit |
| Fruit Ninja | `R` | Restart |
| Fruit Ninja | `ESC` | Quit |

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
- **Pygame**: Game development library for Fruit Ninja
- **NumPy**: Numerical computing (automatically installed with OpenCV)

### Architecture
- **openhouse.py**: Main game launcher with camera selection
- **live_rsp.py**: Rock Paper Scissors with emotion detection
- **draw.py**: Virtual Air Drawing application
- **draw_manager.py**: Drawing object management and transformations
- **hand_tracker.py**: Hand tracking and gesture recognition (2 hands)
- **ui_manager.py**: UI components and help system
- **fruit_ninja.py**: Fruit Ninja game logic and rendering

### Camera Support
- **Resolution Testing**: Automatically tests resolutions from 1920x1080 down to 640x480
- **Multi-Camera**: Supports up to 5 cameras (indices 0-4)
- **Fallback**: Graceful degradation if preferred resolution isn't available
- **Mirroring**: Camera feed mirrored for natural interaction

### Performance
- **FPS**: Typically 20-30 FPS depending on camera and system
- **Latency**: Near real-time gesture and emotion detection
- **Memory**: Approximately 300-600MB RAM usage
- **Two-Hand Tracking**: Simultaneous tracking of both hands for advanced gestures

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
├── openhouse.py          # Game launcher (START HERE!)
├── live_rsp.py           # Rock Paper Scissors game
├── draw.py               # Virtual Air Drawing
├── draw_manager.py       # Drawing object management
├── hand_tracker.py       # Hand tracking (2 hands)
├── ui_manager.py         # UI components
├── fruit_ninja.py        # Fruit Ninja game
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── README_DRAW.md       # Drawing app documentation
├── hand_landmarker.task # MediaPipe hand model
├── Sound/               # Game sound effects
├── fruit_images/        # Fruit and bomb images
└── game_bg/            # Background images
```

### Key Functions
- **openhouse.py**: `GameLauncher` - Main menu and camera selection
- **live_rsp.py**: `detect_emotion()`, `get_hand_gesture()` - RPS game logic
- **draw.py**: `VirtualDrawingApp` - Main drawing application
- **hand_tracker.py**: `HandTracker` - Two-hand gesture recognition
- **draw_manager.py**: `DrawManager` - Object management and transformations
- **fruit_ninja.py**: `FruitNinjaGame` - Fruit slicing game logic

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

## 🎯 Project Highlights

- **3 Games in 1**: Rock Paper Scissors, Virtual Air Drawing, and Fruit Ninja
- **Advanced Hand Tracking**: Two-hand simultaneous tracking with gesture recognition
- **Intuitive Controls**: Natural hand gestures for all interactions
- **Real-time Processing**: Smooth 20-30 FPS performance
- **Professional UI**: Clean interface with help overlays
- **Educational**: Learn computer vision, hand tracking, and game development

## 🙏 Acknowledgments

- **MediaPipe**: Google's MediaPipe framework for hand and face landmark detection
- **OpenCV**: Open Source Computer Vision Library
- **Pygame**: Simple game development library
- **Community**: Thanks to all contributors and testers

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are correctly installed
3. Verify camera permissions and availability
4. Test with different lighting conditions

---

**Enjoy playing Rock Paper Scissors with AI! 🎮✨**
