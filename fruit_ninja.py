"""
Fruit Ninja Game - Finger Slicing Edition
A hand tracking game using MediaPipe where you slice falling fruits with your finger!

Controls:
- Use your INDEX FINGER to slice fruits
- Press 'Q' to quit
- Press 'R' to restart game
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN for faster startup

import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math
import pygame

# ============================================================================
# SOUND INITIALIZATION
# ============================================================================

# Initialize pygame mixer for sound
pygame.mixer.init()

# Sound paths
SOUND_DIR = "Sound"
SLICE_SOUNDS = [
    pygame.mixer.Sound(os.path.join(SOUND_DIR, "Sword-swipe-1.wav")),
    pygame.mixer.Sound(os.path.join(SOUND_DIR, "Sword-swipe-2.wav")),
    pygame.mixer.Sound(os.path.join(SOUND_DIR, "Sword-swipe-3.wav")),
]
IMPACT_SOUNDS = [
    pygame.mixer.Sound(os.path.join(SOUND_DIR, "Splatter-Medium-1.wav")),
    pygame.mixer.Sound(os.path.join(SOUND_DIR, "Splatter-Medium-2.wav")),
]
LOSE_LIFE_SOUND = pygame.mixer.Sound(os.path.join(SOUND_DIR, "Throw-bomb.wav"))
GAME_OVER_SOUND = pygame.mixer.Sound(os.path.join(SOUND_DIR, "Game-over.wav"))
GAME_START_SOUND = pygame.mixer.Sound(os.path.join(SOUND_DIR, "Game-start.wav"))
COMBO_SOUNDS = [
    pygame.mixer.Sound(os.path.join(SOUND_DIR, "combo-1.wav")),
    pygame.mixer.Sound(os.path.join(SOUND_DIR, "combo-2.wav")),
    pygame.mixer.Sound(os.path.join(SOUND_DIR, "combo-3.wav")),
]

# Set volume levels
for sound in SLICE_SOUNDS + IMPACT_SOUNDS:
    sound.set_volume(0.3)
LOSE_LIFE_SOUND.set_volume(0.5)
GAME_OVER_SOUND.set_volume(0.6)
GAME_START_SOUND.set_volume(0.4)
for sound in COMBO_SOUNDS:
    sound.set_volume(0.5)

# ============================================================================
# GAME CONFIGURATION
# ============================================================================

# Display settings
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Game settings
FRUIT_SIZE = 100  # Size of fruit sprites (increased for better visibility)
GRAVITY = 0.3    # Gravity acceleration (lower for smoother fall)
SPAWN_RATE = 1.2  # Seconds between fruit spawns
SLICE_TRAIL_LENGTH = 15  # Length of finger trail (increased for fast motion)

# Fruit types with colors (BGR format)
FRUITS = [
    {
        "name": "Apple", 
        "color": (0, 0, 255), 
        "points": 10,
        "size_range": (60, 100)
    },
    {
        "name": "Grape", 
        "color": (128, 0, 128), 
        "points": 15,
        "size_range": (50, 85)
    },
    {
        "name": "Orange", 
        "color": (0, 165, 255), 
        "points": 10,
        "size_range": (60, 95)
    },
    {
        "name": "Watermelon", 
        "color": (0, 255, 0), 
        "points": 25,
        "size_range": (80, 130)
    },
]

# Dictionary to store loaded fruit images
FRUIT_IMAGES = {}
FRUIT_SLICED_IMAGES = {}

# ============================================================================
# MEDIAPIPE INITIALIZATION
# ============================================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=3,  # Detect up to 3 hands for multiplayer
    min_detection_confidence=0.5,  # Lower to detect partial hands
    min_tracking_confidence=0.3,   # Lower for tracking with partial visibility
    model_complexity=0,            # Faster model for better frame rate
    static_image_mode=False        # Better for video tracking
)

# ============================================================================
# IMAGE LOADING FUNCTIONS
# ============================================================================

def load_fruit_images():
    """Load fruit images from the local 'fruit_images' folder.

    Expected file names (case-insensitive) inside ./fruit_images/:
      - <fruitname>.png or <fruitname>_whole.png  (whole fruit)
      - <fruitname>_sliced.png                   (single sliced image that will be split)
      - <fruitname>_left.png and <fruitname>_right.png (optional separate halves)

    If a sliced image is not present, the whole image will be reused to create halves.
    """
    # Create images directory if it doesn't exist
    img_dir = "fruit_images"
    if not os.path.exists(img_dir):
        print(f"[WARN] '{img_dir}' folder not found. Make sure you placed your fruit images in {img_dir}/")
        return

    for fruit in FRUITS:
        name = fruit["name"]
        key = name
        fname = name.lower()

        # Candidate whole filenames (support PNG, JPG, JPEG)
        whole_candidates = [
            f"{fname}.png", f"{fname}_whole.png", f"{fname}-whole.png",
            f"{fname}.jpg", f"{fname}_whole.jpg", f"{fname}-whole.jpg",
            f"{fname}.jpeg", f"{fname}_whole.jpeg", f"{fname}-whole.jpeg"
        ]
        whole_path = None
        for p in whole_candidates:
            path = os.path.join(img_dir, p)
            if os.path.exists(path):
                whole_path = path
                break

        whole_img = None
        if whole_path:
            whole_img = cv2.imread(whole_path, cv2.IMREAD_UNCHANGED)
        else:
            print(f"[WARN] Whole image for '{name}' not found in {img_dir}/. Expected one of: {whole_candidates}")

        # Candidate sliced filenames (support PNG, JPG, JPEG)
        sliced_candidates = [f"{fname}_sliced.png", f"{fname}_sliced.jpg", f"{fname}_sliced.jpeg"]
        sliced_single = None
        for p in sliced_candidates:
            path = os.path.join(img_dir, p)
            if os.path.exists(path):
                sliced_single = path
                break
        
        left_candidates = [f"{fname}_left.png", f"{fname}_left.jpg", f"{fname}_left.jpeg"]
        left_path = None
        for p in left_candidates:
            path = os.path.join(img_dir, p)
            if os.path.exists(path):
                left_path = path
                break
                
        right_candidates = [f"{fname}_right.png", f"{fname}_right.jpg", f"{fname}_right.jpeg"]
        right_path = None
        for p in right_candidates:
            path = os.path.join(img_dir, p)
            if os.path.exists(path):
                right_path = path
                break

        sliced_img = None
        if sliced_single:
            sliced_img = cv2.imread(sliced_single, cv2.IMREAD_UNCHANGED)
        elif left_path and right_path:
            left = cv2.imread(left_path, cv2.IMREAD_UNCHANGED)
            right = cv2.imread(right_path, cv2.IMREAD_UNCHANGED)
            if left is not None and right is not None:
                # combine horizontally
                try:
                    sliced_img = np.concatenate([left, right], axis=1)
                except Exception:
                    sliced_img = None

        # If sliced not found but whole exists, use whole as sliced fallback (will be split)
        if sliced_img is None and whole_img is not None:
            sliced_img = whole_img.copy()

        # Resize and ensure alpha for whole image
        if whole_img is not None:
            max_size = fruit.get("size_range", (60, 100))[1]
            h, w = whole_img.shape[:2]
            scale = max_size / max(h, w) if max(h, w) > 0 else 1.0
            new_w, new_h = int(w * scale), int(h * scale)
            whole_img = cv2.resize(whole_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if whole_img.ndim == 2:
                whole_img = cv2.cvtColor(whole_img, cv2.COLOR_GRAY2BGRA)
            elif whole_img.shape[2] == 3:
                alpha = np.ones((whole_img.shape[0], whole_img.shape[1], 1), dtype=whole_img.dtype) * 255
                whole_img = np.concatenate([whole_img, alpha], axis=2)
            FRUIT_IMAGES[key] = whole_img
            print(f"Loaded whole image for {name} ({new_w}x{new_h})")

        # Resize and ensure alpha for sliced image
        if sliced_img is not None:
            max_size = fruit.get("size_range", (60, 100))[1]
            h, w = sliced_img.shape[:2]
            scale = max_size / max(h, w) if max(h, w) > 0 else 1.0
            new_w, new_h = int(w * scale), int(h * scale)
            sliced_img = cv2.resize(sliced_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if sliced_img.ndim == 2:
                sliced_img = cv2.cvtColor(sliced_img, cv2.COLOR_GRAY2BGRA)
            elif sliced_img.shape[2] == 3:
                alpha = np.ones((sliced_img.shape[0], sliced_img.shape[1], 1), dtype=sliced_img.dtype) * 255
                sliced_img = np.concatenate([sliced_img, alpha], axis=2)
            FRUIT_SLICED_IMAGES[key] = sliced_img
            print(f"Loaded sliced image for {name} ({new_w}x{new_h})")

# ============================================================================
# CAMERA SELECTION FUNCTIONS
# ============================================================================

def find_available_cameras():
    """Find all available camera indices"""
    available_cameras = []
    max_cameras_to_check = 5
    
    print("\nScanning for available cameras...")
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                available_cameras.append({
                    'index': i,
                    'width': width,
                    'height': height
                })
                print(f"  ✓ Camera {i}: {width}x{height}")
            cap.release()
        else:
            print(f"  ✗ Camera {i}: Not available")
    
    return available_cameras

def choose_camera():
    """Let user choose from available cameras or auto-select camera 1 if 3+ cameras available"""
    available_cameras = find_available_cameras()
    
    if not available_cameras:
        print("❌ No cameras found! Please check your camera connections.")
        exit()
    
    # Auto-select camera 1 if 3 or more cameras available
    if len(available_cameras) >= 3:
        default_camera = 1
        # Check if camera 1 is available
        if any(cam['index'] == default_camera for cam in available_cameras):
            print(f"\n✅ Auto-selected Camera {default_camera} (3+ cameras detected)")
            return default_camera
        else:
            # Fallback to first available camera
            default_camera = available_cameras[0]['index']
            print(f"\n✅ Auto-selected Camera {default_camera} (default)")
            return default_camera
    
    # Manual selection for less than 3 cameras
    print("\n" + "="*50)
    print("📷 AVAILABLE CAMERAS:")
    print("="*50)
    
    for camera in available_cameras:
        print(f"  [{camera['index']}] Camera {camera['index']} - {camera['width']}x{camera['height']}")
    
    print("="*50)
    
    while True:
        try:
            user_input = input(f"Choose camera index (available: {[cam['index'] for cam in available_cameras]}): ").strip()
            selected_index = int(user_input)
            
            if any(cam['index'] == selected_index for cam in available_cameras):
                print(f"✅ Selected Camera {selected_index}")
                return selected_index
            else:
                print(f"❌ Camera {selected_index} is not available. Please choose from: {[cam['index'] for cam in available_cameras]}")
        
        except ValueError:
            print("❌ Please enter a valid number.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            exit()

# ============================================================================
# GAME CLASSES
# ============================================================================

class Fruit:
    """Represents a falling fruit that can be sliced"""
    
    def __init__(self, x, y, vx, vy, fruit_type):
        self.x = x
        self.y = y
        self.vx = vx  # Velocity X
        self.vy = vy  # Velocity Y
        self.fruit_type = fruit_type
        # Random size within fruit's size range
        size_range = fruit_type.get("size_range", (60, 100))
        self.size = random.randint(size_range[0], size_range[1])
        self.sliced = False
        self.slice_time = 0
        self.slice_angle = 0  # Angle of the slice for rotation
        
        # Per-half physics for sliced fruit
        self.left_half = {"x": x, "y": y, "vx": 0, "vy": 0, "rotation": 0, "angular_velocity": 0}
        self.right_half = {"x": x, "y": y, "vx": 0, "vy": 0, "rotation": 0, "angular_velocity": 0}
        
    def update(self):
        """Update fruit position with gravity"""
        if not self.sliced:
            # Update whole fruit
            self.x += self.vx
            self.y += self.vy
            self.vy += GRAVITY  # Apply gravity
            
            # Keep fruits within horizontal bounds (bounce off edges)
            if self.x < FRUIT_SIZE:
                self.x = FRUIT_SIZE
                self.vx = abs(self.vx) * 0.8  # Bounce back with reduced velocity
            elif self.x > WINDOW_WIDTH - FRUIT_SIZE:
                self.x = WINDOW_WIDTH - FRUIT_SIZE
                self.vx = -abs(self.vx) * 0.8  # Bounce back with reduced velocity
        else:
            # Update each half separately with physics
            for half in [self.left_half, self.right_half]:
                half["x"] += half["vx"]
                half["y"] += half["vy"]
                half["vy"] += GRAVITY  # Apply gravity to each half
                half["rotation"] += half["angular_velocity"]  # Update rotation
                half["angular_velocity"] *= 0.98  # Slight air resistance on rotation
        
    def is_off_screen(self, width, height):
        """Check if fruit has fallen off screen"""
        if not self.sliced:
            return self.y > height + self.size or self.x < -self.size or self.x > width + self.size
        else:
            # Check if both halves are off screen
            left_off = self.left_half["y"] > height + self.size
            right_off = self.right_half["y"] > height + self.size
            return left_off and right_off
    
    def _rotate_image(self, img, angle):
        """Rotate image by angle (degrees) while preserving transparency"""
        if img is None or img.size == 0:
            return img
            
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding dimensions
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix for new center
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]
        
        # Rotate with transparency
        rotated = cv2.warpAffine(img, matrix, (new_w, new_h), 
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
        return rotated
    
    def _overlay_image_centered(self, frame, img):
        """Overlay image centered at fruit position"""
        x1 = int(self.x - img.shape[1] // 2)
        y1 = int(self.y - img.shape[0] // 2)
        self._overlay_image_at(frame, img, x1, y1)
    
    def _overlay_image_at(self, frame, img, x1, y1):
        """Overlay image at specific position"""
        if img is None or img.size == 0:
            return
            
        x2 = x1 + img.shape[1]
        y2 = y1 + img.shape[0]
        
        # Check boundaries
        if x1 >= frame.shape[1] or y1 >= frame.shape[0] or x2 <= 0 or y2 <= 0:
            return
        
        # Adjust for boundaries
        img_x1 = max(0, -x1)
        img_y1 = max(0, -y1)
        img_x2 = img.shape[1] - max(0, x2 - frame.shape[1])
        img_y2 = img.shape[0] - max(0, y2 - frame.shape[0])
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        # Get the region of interest
        roi = frame[y1:y2, x1:x2]
        img_crop = img[img_y1:img_y2, img_x1:img_x2]
        
        # Blend using alpha channel
        if img_crop.shape[2] == 4 and roi.shape[:2] == img_crop.shape[:2]:
            alpha = img_crop[:, :, 3] / 255.0
            alpha = alpha[:, :, np.newaxis]
            frame[y1:y2, x1:x2] = (alpha * img_crop[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)
    
    def _overlay_image(self, frame, img):
        """Overlay image with transparency at fruit position"""
        # Calculate position (top-left corner)
        x1 = int(self.x - self.size // 2)
        y1 = int(self.y - self.size // 2)
        x2 = x1 + self.size
        y2 = y1 + self.size
        
        # Check boundaries
        if x1 >= frame.shape[1] or y1 >= frame.shape[0] or x2 <= 0 or y2 <= 0:
            return
        
        # Adjust for boundaries
        img_x1 = max(0, -x1)
        img_y1 = max(0, -y1)
        img_x2 = self.size - max(0, x2 - frame.shape[1])
        img_y2 = self.size - max(0, y2 - frame.shape[0])
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        # Get the region of interest
        roi = frame[y1:y2, x1:x2]
        img_crop = img[img_y1:img_y2, img_x1:img_x2]
        
        # Blend using alpha channel
        if img_crop.shape[2] == 4:
            alpha = img_crop[:, :, 3] / 255.0
            alpha = alpha[:, :, np.newaxis]
            
            # Ensure ROI and image crop have the same dimensions
            if roi.shape[:2] == img_crop.shape[:2]:
                frame[y1:y2, x1:x2] = (alpha * img_crop[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)
    
    def check_collision(self, finger_x, finger_y, finger_prev_x, finger_prev_y):
        """Check if finger sliced through the fruit"""
        # Calculate distance from fruit center to line segment (finger movement)
        if finger_prev_x is None or finger_prev_y is None:
            return False
            
        # Check collision with larger detection radius for fast movements
        detection_radius = self.size / 1.5  # Increased detection area
        
        # Simple circle-line collision detection
        # Check if finger path passes through fruit circle
        dist = self._point_to_line_distance(
            self.x, self.y, 
            finger_prev_x, finger_prev_y, 
            finger_x, finger_y
        )
        
        return dist < detection_radius
    
    def _point_to_line_distance(self, px, py, x1, y1, x2, y2):
        """Calculate distance from point to line segment"""
        # Vector from point 1 to point 2
        dx = x2 - x1
        dy = y2 - y1
        
        if dx == 0 and dy == 0:
            # Line segment is a point
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        
        # Calculate projection of point onto line
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        
        # Find closest point on line segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Return distance to closest point
        return math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)
    
    def draw(self, frame):
        """Draw the fruit on the frame"""
        fruit_name = self.fruit_type["name"]
        
        if not self.sliced:
            # Draw whole fruit
            if fruit_name in FRUIT_IMAGES:
                # Resize image to match this fruit's size
                img = FRUIT_IMAGES[fruit_name]
                scaled_img = cv2.resize(img, (self.size, self.size))
                self._overlay_image_centered(frame, scaled_img)
            else:
                # Fallback to circle
                color = self.fruit_type["color"]
                cv2.circle(frame, (int(self.x), int(self.y)), self.size // 2, color, -1)
                cv2.circle(frame, (int(self.x), int(self.y)), self.size // 2, (255, 255, 255), 3)
                cv2.putText(frame, fruit_name[:1], 
                           (int(self.x - 15), int(self.y + 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            # Draw sliced fruit using per-half physics
            if fruit_name in FRUIT_SLICED_IMAGES:
                # Draw sliced image with realistic physics
                sliced_img = FRUIT_SLICED_IMAGES[fruit_name]
                scaled_sliced = cv2.resize(sliced_img, (self.size, self.size))
                
                # Split sliced image into two halves
                half_width = scaled_sliced.shape[1] // 2
                
                # Left half - use its physics
                left_half_img = scaled_sliced[:, :half_width]
                left_rotated = self._rotate_image(left_half_img, self.left_half["rotation"])
                self._overlay_image_at(frame, left_rotated, 
                                      int(self.left_half["x"] - left_rotated.shape[1] // 2), 
                                      int(self.left_half["y"] - left_rotated.shape[0] // 2))
                
                # Right half - use its physics
                right_half_img = scaled_sliced[:, half_width:]
                right_rotated = self._rotate_image(right_half_img, self.right_half["rotation"])
                self._overlay_image_at(frame, right_rotated, 
                                       int(self.right_half["x"] - right_rotated.shape[1] // 2), 
                                       int(self.right_half["y"] - right_rotated.shape[0] // 2))
            else:
                # Fallback to circles with physics
                color = self.fruit_type["color"]
                cv2.circle(frame, (int(self.left_half["x"]), int(self.left_half["y"])), 
                          self.size // 3, color, -1)
                cv2.circle(frame, (int(self.right_half["x"]), int(self.right_half["y"])), 
                          self.size // 3, color, -1)

class Game:
    """Main game logic controller"""
    
    def __init__(self):
        self.fruits = []
        self.score = 0
        self.lives = 3
        self.last_spawn_time = time.time()
        self.start_time = time.time()  # Track game start time
        self.final_time = 0  # Store time when game ends
        self.game_over = False
        self.finger_trails = []  # Store finger trails for each hand (list of lists)
        self.finger_positions = []  # Store current finger positions for all hands
        self.prev_finger_positions = []  # Store previous finger positions for all hands
        self.difficulty_level = 1  # Track current difficulty
        self.combo = 0  # Track consecutive slices
        self.last_slice_time = 0  # Track last slice for combo
        
        # Play game start sound
        GAME_START_SOUND.play()
        
    def get_difficulty_multiplier(self):
        """Calculate difficulty multiplier based on elapsed time"""
        elapsed_time = time.time() - self.start_time
        # Increase difficulty every 20 seconds, max 3x difficulty
        multiplier = min(1 + (elapsed_time / 20) * 0.5, 3.0)
        self.difficulty_level = int(multiplier * 2)  # For display (1-6)
        return multiplier
    
    def spawn_fruit(self):
        """Spawn a new fruit at the top with downward velocity"""
        difficulty = self.get_difficulty_multiplier()
        
        # Random spawn position at top of screen - more centered to prevent edge spawns
        x = random.randint(150, WINDOW_WIDTH - 150)
        y = -FRUIT_SIZE  # Start above screen
        
        # Random velocities - REDUCED horizontal velocity to keep fruits visible
        # Maximum horizontal velocity reduced to prevent fruits going off-screen
        max_horizontal = min(2.0, 2.0 / difficulty)  # Slower horizontal at higher difficulty
        vx = random.uniform(-max_horizontal, max_horizontal)
        vy = random.uniform(2, 5) * difficulty   # Initial downward velocity increases
        
        # Random fruit type
        fruit_type = random.choice(FRUITS)
        
        fruit = Fruit(x, y, vx, vy, fruit_type)
        self.fruits.append(fruit)
    
    def update(self):
        """Update all game objects"""
        if self.game_over:
            return
        
        # Spawn new fruits - spawn rate increases with difficulty
        current_time = time.time()
        difficulty = self.get_difficulty_multiplier()
        # Spawn faster as difficulty increases (min 0.4 seconds between spawns)
        adjusted_spawn_rate = max(SPAWN_RATE / difficulty, 0.4)
        
        if current_time - self.last_spawn_time > adjusted_spawn_rate:
            self.spawn_fruit()
            self.last_spawn_time = current_time
        
        # Update fruits
        fruits_to_remove = []
        for fruit in self.fruits:
            fruit.update()
            
            # Check if fruit was sliced - check against all hands' trails
            if not fruit.sliced:
                sliced = False
                for trail in self.finger_trails:
                    if len(trail) >= 2:
                        # Check the last few segments of the finger trail
                        check_segments = min(5, len(trail) - 1)  # Check last 5 segments
                        for i in range(len(trail) - 1, len(trail) - check_segments - 1, -1):
                            if i > 0:
                                x1, y1 = trail[i-1]
                                x2, y2 = trail[i]
                                if fruit.check_collision(x2, y2, x1, y1):
                                    sliced = True
                                    break
                        if sliced:
                            break
                
                if sliced:
                            fruit.sliced = True
                            fruit.slice_time = time.time()
                            self.score += fruit.fruit_type["points"]
                            
                            # Initialize per-half physics for realistic slicing
                            slice_force = 8  # Force applied to halves
                            # Left half flies left and rotates counter-clockwise
                            fruit.left_half = {
                                "x": fruit.x,
                                "y": fruit.y,
                                "vx": fruit.vx - slice_force - random.uniform(1, 3),
                                "vy": fruit.vy - random.uniform(2, 4),
                                "rotation": 0,
                                "angular_velocity": random.uniform(-8, -3)
                            }
                            # Right half flies right and rotates clockwise
                            fruit.right_half = {
                                "x": fruit.x,
                                "y": fruit.y,
                                "vx": fruit.vx + slice_force + random.uniform(1, 3),
                                "vy": fruit.vy - random.uniform(2, 4),
                                "rotation": 0,
                                "angular_velocity": random.uniform(3, 8)
                            }
                            
                            # Track combo (fruits sliced within 1 second of each other)
                            current_time = time.time()
                            if current_time - self.last_slice_time < 1.0:
                                self.combo += 1
                                # Play combo sound for 3+ combo
                                if self.combo >= 3 and self.combo <= 5:
                                    COMBO_SOUNDS[min(self.combo - 3, 2)].play()
                            else:
                                self.combo = 1
                            self.last_slice_time = current_time
                            
                            # Play slice sound
                            random.choice(SLICE_SOUNDS).play()
                            # Play impact sound slightly delayed
                            random.choice(IMPACT_SOUNDS).play()
                            break
            
            # Remove off-screen fruits
            if fruit.is_off_screen(WINDOW_WIDTH, WINDOW_HEIGHT):
                if not fruit.sliced:
                    # Lost a life if fruit wasn't sliced
                    self.lives -= 1
                    self.combo = 0  # Reset combo on miss
                    LOSE_LIFE_SOUND.play()  # Play lose life sound
                    if self.lives <= 0:
                        self.game_over = True
                        self.final_time = int(time.time() - self.start_time)  # Capture final time
                        GAME_OVER_SOUND.play()  # Play game over sound
                if fruit not in fruits_to_remove:  # Avoid duplicates
                    fruits_to_remove.append(fruit)
            
            # Remove sliced fruits after animation
            elif fruit.sliced and time.time() - fruit.slice_time > 0.5:  # Use elif to prevent double-add
                fruits_to_remove.append(fruit)
        
        # Remove marked fruits
        for fruit in fruits_to_remove:
            if fruit in self.fruits:  # Safety check
                self.fruits.remove(fruit)
    
    def draw(self, frame):
        """Draw all game elements"""
        # Draw fruits
        for fruit in self.fruits:
            fruit.draw(frame)
        
        # Draw finger trails for all hands
        trail_colors = [(0, 255, 255), (255, 0, 255), (0, 255, 0)]  # Cyan, Magenta, Green
        for hand_idx, trail in enumerate(self.finger_trails):
            if len(trail) > 1:
                color = trail_colors[hand_idx % len(trail_colors)]
                for i in range(1, len(trail)):
                    # Fade trail effect
                    thickness = max(1, int((i / len(trail)) * 5))
                    cv2.line(frame, trail[i-1], trail[i], color, thickness)
        
        # Draw UI - Score and Difficulty
        cv2.rectangle(frame, (10, 10), (300, 130), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 130), (0, 255, 255), 3)
        cv2.putText(frame, f"SCORE: {self.score}", (20, 45),
                   cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 255), 2)
        
        # Draw difficulty level with color coding
        diff_color = (0, 255, 0) if self.difficulty_level <= 2 else (0, 255, 255) if self.difficulty_level <= 4 else (0, 0, 255)
        cv2.putText(frame, f"LEVEL: {self.difficulty_level}", (20, 80),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, diff_color, 2)
        
        # Draw lives (hearts)
        heart_x = 20
        for i in range(self.lives):
            cv2.circle(frame, (heart_x + i * 40, 110), 12, (0, 0, 255), -1)
        
        # Draw elapsed time (top right)
        if not self.game_over:
            elapsed = int(time.time() - self.start_time)
            minutes = elapsed // 60
            seconds = elapsed % 60
            cv2.rectangle(frame, (WINDOW_WIDTH - 200, 10), (WINDOW_WIDTH - 10, 70), (0, 0, 0), -1)
            cv2.rectangle(frame, (WINDOW_WIDTH - 200, 10), (WINDOW_WIDTH - 10, 70), (255, 255, 0), 3)
            cv2.putText(frame, f"TIME: {minutes:02d}:{seconds:02d}", 
                       (WINDOW_WIDTH - 190, 50),
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 0), 2)
        
        # Draw combo (center screen) if combo >= 3
        if self.combo >= 3 and not self.game_over:
            combo_color = (0, 255, 0) if self.combo < 5 else (0, 165, 255) if self.combo < 10 else (0, 0, 255)
            cv2.putText(frame, f"COMBO x{self.combo}!", 
                       (WINDOW_WIDTH // 2 - 150, 100),
                       cv2.FONT_HERSHEY_DUPLEX, 2.0, combo_color, 4)
        
        # Game over screen
        if self.game_over:
            # Semi-transparent overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (WINDOW_WIDTH, WINDOW_HEIGHT), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Game over text - use captured final_time instead of calculating
            minutes = self.final_time // 60
            seconds = self.final_time % 60
            
            cv2.putText(frame, "GAME OVER!", 
                       (WINDOW_WIDTH // 2 - 200, WINDOW_HEIGHT // 2 - 100),
                       cv2.FONT_HERSHEY_DUPLEX, 2.5, (0, 0, 255), 4)
            cv2.putText(frame, f"Final Score: {self.score}", 
                       (WINDOW_WIDTH // 2 - 180, WINDOW_HEIGHT // 2 - 10),
                       cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 255, 255), 3)
            cv2.putText(frame, f"Time Survived: {minutes:02d}:{seconds:02d}", 
                       (WINDOW_WIDTH // 2 - 180, WINDOW_HEIGHT // 2 + 40),
                       cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 0), 3)
            cv2.putText(frame, f"Max Level: {self.difficulty_level}", 
                       (WINDOW_WIDTH // 2 - 130, WINDOW_HEIGHT // 2 + 85),
                       cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'R' to Restart or 'Q' to Quit", 
                       (WINDOW_WIDTH // 2 - 280, WINDOW_HEIGHT // 2 + 140),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    def update_finger_positions(self, positions):
        """Update finger tracking positions for multiple hands"""
        self.prev_finger_positions = self.finger_positions.copy()
        self.finger_positions = positions
        
        # Update trails for each hand
        while len(self.finger_trails) < len(positions):
            self.finger_trails.append([])
        
        # Add current positions to trails
        for i, (x, y) in enumerate(positions):
            if i < len(self.finger_trails):
                self.finger_trails[i].append((x, y))
                if len(self.finger_trails[i]) > SLICE_TRAIL_LENGTH:
                    self.finger_trails[i].pop(0)
        
        # Remove extra trails if fewer hands detected
        if len(self.finger_trails) > len(positions):
            self.finger_trails = self.finger_trails[:len(positions)]
    
    def reset(self):
        """Reset game state"""
        self.fruits = []
        self.score = 0
        self.lives = 5
        self.last_spawn_time = time.time()
        self.start_time = time.time()  # Reset timer
        self.final_time = 0  # Reset final time
        self.game_over = False
        self.finger_trails = []
        self.finger_positions = []
        self.prev_finger_positions = []
        self.difficulty_level = 1  # Reset difficulty level
        self.combo = 0  # Reset combo

# ============================================================================
# MAIN GAME LOOP
# ============================================================================

def main():
    """Main application entry point"""
    
    print("=" * 60)
    print("Fruit Ninja - Finger Slicing Edition")
    print("=" * 60)
    print("\n[LOADING RESOURCES]")
    print("Loading fruit images...")
    load_fruit_images()
    print("✓ Images loaded successfully!\n")
    
    print("\n[INSTRUCTIONS]")
    print("- Use your INDEX FINGER to slice fruits")
    print("- Keep your FULL HAND visible to the camera")
    print("- Don't let fruits fall off screen!")
    print("- You have 3 lives")
    print("- Press 'Q' to quit, 'R' to restart, 'B' or ESC to go back")
    
    # Get camera index (from command line argument or choose manually)
    import sys
    if len(sys.argv) > 1:
        # Camera index provided as command line argument
        camera_index = int(sys.argv[1])
        print(f"\n[USING CAMERA {camera_index} FROM LAUNCHER]\n")
    else:
        # Let user choose camera manually
        camera_index = choose_camera()
    
    print("\n[STARTING GAME...]\n")
    
    # Initialize webcam with selected camera
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)  # Request higher frame rate
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_index}!")
        return
    
    # Get actual camera resolution and FPS
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"📷 Camera resolution: {actual_width}x{actual_height} @ {actual_fps}fps")
    
    # Warm up camera
    print("🔥 Warming up camera...")
    for i in range(5):
        cap.read()
    print("✅ Camera ready!")
    
    # Create game window
    cv2.namedWindow('Fruit Ninja', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Fruit Ninja', WINDOW_WIDTH, WINDOW_HEIGHT)
    
    # Initialize game
    game = Game()
    
    print("[OK] Game started! Start slicing fruits!\n")
    
    # Main game loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame")
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Detect hands and get index finger tip positions
        hand_detected = False
        finger_positions = []
        if results.multi_hand_landmarks:
            hand_detected = True
            finger_colors = [(0, 255, 255), (255, 0, 255), (0, 255, 0)]  # Cyan, Magenta, Green
            
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get index finger tip (landmark 8)
                index_tip = hand_landmarks.landmark[8]
                finger_x = int(index_tip.x * WINDOW_WIDTH)
                finger_y = int(index_tip.y * WINDOW_HEIGHT)
                
                finger_positions.append((finger_x, finger_y))
                
                # Draw hand landmarks with different colors
                color = finger_colors[hand_idx % len(finger_colors)]
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=color, thickness=2)
                )
                
                # Highlight index finger with hand-specific color
                cv2.circle(frame, (finger_x, finger_y), 15, color, -1)
                cv2.circle(frame, (finger_x, finger_y), 15, (255, 255, 255), 3)
            
            # Update game with all finger positions
            game.update_finger_positions(finger_positions)
        else:
            # Clear trails when no hands detected
            game.update_finger_positions([])
            
            # Show warning when hand is not detected
            if not game.game_over:
                cv2.putText(frame, "HAND NOT DETECTED!", 
                           (WINDOW_WIDTH // 2 - 200, WINDOW_HEIGHT - 50),
                           cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(frame, "Keep your hand visible to the camera", 
                           (WINDOW_WIDTH // 2 - 250, WINDOW_HEIGHT - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        # Update game state
        game.update()
        
        # Draw game elements
        game.draw(frame)
        
        # Display frame
        cv2.imshow('Fruit Ninja', frame)
        
        # Draw back button
        back_button_x, back_button_y = 20, 20
        back_button_w, back_button_h = 120, 50
        cv2.rectangle(frame, (back_button_x, back_button_y),
                     (back_button_x + back_button_w, back_button_y + back_button_h),
                     (50, 50, 50), -1)
        cv2.rectangle(frame, (back_button_x, back_button_y),
                     (back_button_x + back_button_w, back_button_y + back_button_h),
                     (255, 255, 255), 2)
        cv2.putText(frame, "< BACK", (back_button_x + 10, back_button_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[QUIT] Thanks for playing!")
            break
        elif key == ord('b') or key == 27:  # B or ESC for back
            print("\n[BACK] Returning to menu...")
            break
        elif key == ord('r'):
            print("\n[RESTART] Starting new game...")
            game.reset()
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    
    print(f"\n[FINAL SCORE] {game.score} points!")
    print("=" * 60)

if __name__ == "__main__":
    main()
