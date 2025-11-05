# Virtual Air Drawing Application

An AI-powered virtual whiteboard that lets you draw and manipulate 2D and 3D shapes in mid-air using hand gestures captured from your webcam.

## 🎯 Features

### Hand Gesture Controls
- **Index Finger Up** → Draw mode (pen tool)
- **Pinch (Thumb + Index)** → Select and move objects
- **Two-Finger Pinch** → Toggle between 2D and 3D mode
- **Three Fingers Up** → Open shape menu and color palette
- **Open Palm** → Stop drawing/action

### Drawing Tools
- **2D Shapes**: Lines (free drawing), Circles, Rectangles, Triangles
- **3D Shapes**: Cubes, Pyramids, Spheres (with auto-rotation)
- **Object Manipulation**: Move, resize, and rotate shapes
- **Color Palette**: 8 colors to choose from

### Interactive Features
- Real-time hand tracking with MediaPipe
- Smooth gesture recognition with cooldown
- Visual feedback for gestures
- FPS counter for performance monitoring
- Undo/redo functionality
- Clear canvas option

## 📦 Requirements

```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.21.0
```

## 🚀 Installation

1. Install Python 3.8 or higher

2. Install required packages:
```bash
pip install opencv-python mediapipe numpy
```

3. Run the application:
```bash
python draw.py
```

## 🎮 Controls

### Gestures
| Gesture | Action |
|---------|--------|
| 1 Finger Up | Draw/Place shapes |
| Pinch | Select and move objects |
| 2-Finger Pinch | Toggle 2D/3D mode |
| 3 Fingers Up | Show shape & color menu |
| Open Palm | Stop current action |

### Keyboard
| Key | Action |
|-----|--------|
| `C` | Clear canvas |
| `H` | Toggle help overlay |
| `U` | Undo last action |
| `Q` | Quit application |

## 📁 Project Structure

```
├── draw.py           # Main application entry point
├── hand_tracker.py   # Hand detection and gesture recognition
├── draw_manager.py   # 2D drawing logic and object management
├── shape_3d.py       # 3D shape rendering and transformations
├── ui_manager.py     # UI elements and on-screen interface
└── README_DRAW.md    # This file
```

## 🔧 How It Works

### Hand Tracking (hand_tracker.py)
- Uses MediaPipe Hands for real-time hand landmark detection
- Recognizes gestures based on finger positions and distances
- Provides smooth gesture transitions with cooldown mechanism

### 2D Drawing (draw_manager.py)
- Object-oriented design for drawable shapes
- Each shape is a separate object with position, size, color
- Supports selection, movement, and transformation
- Maintains drawing history for undo functionality

### 3D Rendering (shape_3d.py)
- Perspective projection for 3D-to-2D conversion
- Rotation matrices for shape transformations
- Wireframe rendering with visible edges
- Auto-rotation for dynamic visualization

### UI System (ui_manager.py)
- Floating buttons and menus
- Color palette with visual selection
- Shape menu for tool switching
- Real-time gesture feedback
- Help overlay with controls

## 🎨 Usage Examples

### Drawing in 2D
1. Show your index finger to enter draw mode
2. Move your finger in the air to draw lines
3. Use 3-finger gesture to open shape menu
4. Select a shape (circle, rectangle, triangle)
5. Point with index finger to place the shape

### Working with 3D Shapes
1. Use 2-finger pinch gesture to toggle to 3D mode
2. Open shape menu (3 fingers) and select cube/pyramid/sphere
3. Point with index finger to place 3D shape
4. Shapes will auto-rotate for better visualization
5. Use pinch gesture to select and move shapes

### Selecting Colors
1. Show 3 fingers to open color palette (bottom of screen)
2. Move your hand over desired color
3. Color will be highlighted when selected
4. Draw with new color

### Moving Objects
1. Make a pinch gesture (thumb + index together)
2. Move to the object you want to select
3. Keep pinching and move your hand to drag the object
4. Open palm to release

## ⚙️ Configuration

You can adjust settings in the respective modules:

### Hand Tracker Settings
```python
# In hand_tracker.py
self.hands = self.mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,  # Adjust detection threshold
    min_tracking_confidence=0.5     # Adjust tracking threshold
)
```

### Drawing Settings
```python
# In draw_manager.py
self.current_color = (0, 255, 0)  # Default color (BGR)
self.thickness = 3                 # Line thickness
```

### 3D Rendering Settings
```python
# In shape_3d.py
self.fov = 500              # Field of view
self.camera_distance = 500  # Camera distance
```

## 🐛 Troubleshooting

### Camera Not Opening
- Check if another application is using the camera
- Try different camera indices in `setup_camera()`
- Ensure camera permissions are granted

### Poor Hand Detection
- Ensure good lighting conditions
- Keep hand at medium distance from camera
- Make sure full hand is visible, not just fingertips

### Low FPS
- Close other resource-intensive applications
- Reduce camera resolution in `__init__`
- Disable auto-rotation in 3D mode

### Gesture Not Recognized
- Gestures have cooldown to prevent jitter
- Make clear, deliberate gestures
- Adjust confidence thresholds in hand_tracker.py

## 🌟 Advanced Features

### Customizing Shapes
Add new shapes by extending `DrawableObject` in `draw_manager.py`:

```python
class MyShape(DrawableObject):
    def __init__(self, position, color=(255, 0, 0)):
        super().__init__('myshape', position, color)
    
    def draw(self, frame):
        # Your drawing code here
        pass
```

### Adding New Gestures
Extend `_recognize_gesture()` in `hand_tracker.py`:

```python
def _recognize_gesture(self):
    # Your gesture recognition logic
    if custom_condition:
        return 'my_gesture'
```

## 📝 Performance Notes

- **Target FPS**: 20-30 FPS (real-time interaction)
- **Hand Tracking**: ~10-15ms per frame
- **Rendering**: 2D faster than 3D mode
- **Memory**: Minimal (~100MB for typical session)

## 🤝 Contributing

Feel free to enhance the application:
- Add new gesture types
- Implement new shape types
- Improve 3D rendering (textures, lighting)
- Add save/load functionality
- Implement gesture-based color mixing

## 📄 License

This project is for educational purposes. Feel free to modify and distribute.

## 🙏 Credits

- **MediaPipe** - Google's hand tracking solution
- **OpenCV** - Computer vision library
- **NumPy** - Numerical computing

---

**Enjoy drawing in the air!** ✨🎨
