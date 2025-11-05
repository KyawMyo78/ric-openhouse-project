"""
Game Launcher - Open House Edition
Choose between Rock Paper Scissors and Fruit Ninja games
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import sys
import subprocess

# Window settings
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
WINDOW_NAME = "Game Launcher - Choose Your Game"

class GameButton:
    """Interactive game button with background image"""
    
    def __init__(self, x, y, width, height, game_name, bg_path, script_path):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.game_name = game_name
        self.script_path = script_path
        self.is_hovered = False
        
        # Load and resize background image
        self.bg_image = self._load_background(bg_path)
        
    def _load_background(self, bg_path):
        """Load and resize background image"""
        if not os.path.exists(bg_path):
            print(f"[WARN] Background not found: {bg_path}")
            # Create a gradient background as fallback
            bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for i in range(self.height):
                color_val = int(50 + (i / self.height) * 100)
                bg[i, :] = [color_val, color_val // 2, color_val // 3]
            return bg
        
        # Load image
        img = cv2.imread(bg_path)
        if img is None:
            print(f"[ERROR] Could not load: {bg_path}")
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Resize to button dimensions
        img = cv2.resize(img, (self.width, self.height))
        return img
    
    def draw(self, frame):
        """Draw button with background"""
        # Create a copy of the background
        button_img = self.bg_image.copy()
        
        # Apply hover effect (brighten)
        if self.is_hovered:
            button_img = cv2.addWeighted(button_img, 0.7, 
                                         np.ones_like(button_img) * 255, 0.3, 0)
            
            # Add white border for hover effect
            cv2.rectangle(button_img, (5, 5), 
                         (self.width - 5, self.height - 5),
                         (255, 255, 255), 8)
        
        # Add game title at the top with shadow
        title_font = cv2.FONT_HERSHEY_DUPLEX
        title_scale = 2.5
        title_thickness = 6
        
        # Get text size for centering
        (text_width, text_height), _ = cv2.getTextSize(
            self.game_name, title_font, title_scale, title_thickness
        )
        
        # Calculate center position
        text_x = (self.width - text_width) // 2
        text_y = 100
        
        # Draw shadow
        cv2.putText(button_img, self.game_name, 
                   (text_x + 4, text_y + 4),
                   title_font, title_scale, (0, 0, 0), title_thickness + 2)
        
        # Draw main text
        cv2.putText(button_img, self.game_name, 
                   (text_x, text_y),
                   title_font, title_scale, (255, 255, 255), title_thickness)
        
        # Add "Click to Play" text at bottom if hovered
        if self.is_hovered:
            play_text = "CLICK TO PLAY"
            play_font = cv2.FONT_HERSHEY_SIMPLEX
            play_scale = 1.2
            play_thickness = 3
            
            (play_width, play_height), _ = cv2.getTextSize(
                play_text, play_font, play_scale, play_thickness
            )
            
            play_x = (self.width - play_width) // 2
            play_y = self.height - 80
            
            # Animated pulsing effect
            import time
            pulse = abs(np.sin(time.time() * 3)) * 0.3 + 0.7
            color = (int(255 * pulse), int(255 * pulse), int(50))
            
            # Draw shadow
            cv2.putText(button_img, play_text,
                       (play_x + 3, play_y + 3),
                       play_font, play_scale, (0, 0, 0), play_thickness + 1)
            
            # Draw main text
            cv2.putText(button_img, play_text,
                       (play_x, play_y),
                       play_font, play_scale, color, play_thickness)
        
        # Place button on frame
        frame[self.y:self.y + self.height, 
              self.x:self.x + self.width] = button_img
    
    def contains(self, x, y):
        """Check if point is inside button"""
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)
    
    def launch_game(self, camera_index):
        """Launch the game script with selected camera"""
        print(f"\n{'='*60}")
        print(f"🎮 Launching {self.game_name}...")
        print(f"📷 Using Camera {camera_index}")
        print(f"⏳ Loading MediaPipe models (this may take 10-30 seconds)...")
        print(f"{'='*60}\n")
        
        try:
            # Close the launcher window temporarily
            cv2.destroyAllWindows()
            
            # Launch the game with camera index as argument (wait for it to finish)
            subprocess.run([sys.executable, self.script_path, str(camera_index)])
            
            # Game has finished, return True to indicate we should continue launcher
            print(f"\n{'='*60}")
            print(f"✅ {self.game_name} closed. Returning to launcher...")
            print(f"{'='*60}\n")
            return True
            
        except Exception as e:
            print(f"[ERROR] Could not launch game: {e}")
            return False


class GameLauncher:
    """Main game launcher application"""
    
    def __init__(self, camera_index):
        self.running = True
        self.camera_index = camera_index
        self.game_to_launch = None  # Flag to track which game to launch
        
        # Create game buttons (side by side, full height)
        button_width = WINDOW_WIDTH // 2
        button_height = WINDOW_HEIGHT
        
        self.buttons = [
            GameButton(
                0, 0, button_width, button_height,
                "Rock Paper Scissors",
                "game_bg/rps_bg.jpg",
                "live_rsp.py"
            ),
            GameButton(
                button_width, 0, button_width, button_height,
                "Fruit Ninja",
                "game_bg/fruitninja_bg.webp",
                "fruit_ninja.py"
            )
        ]
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_MOUSEMOVE:
            # Update hover state for all buttons
            for button in self.buttons:
                button.is_hovered = button.contains(x, y)
                
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Check if any button was clicked
            for button in self.buttons:
                if button.contains(x, y):
                    # Set flag to launch game (don't launch from callback)
                    self.game_to_launch = button
                    return
    
    def run(self):
        """Main loop - keeps running until user quits"""
        print("=" * 60)
        print("GAME LAUNCHER - OPEN HOUSE EDITION")
        print("=" * 60)
        print("Choose your game:")
        print("  - Rock Paper Scissors (Left)")
        print("  - Fruit Ninja (Right)")
        print("\nPress 'Q' to quit")
        print("=" * 60 + "\n")
        
        # Create window
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)
        
        while self.running:
            # Check if a game needs to be launched
            if self.game_to_launch is not None:
                button_to_launch = self.game_to_launch
                self.game_to_launch = None  # Reset flag
                
                # Show loading screen
                loading_frame = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
                loading_frame[:] = [20, 20, 20]
                
                # Loading text
                loading_text = f"Loading {button_to_launch.game_name}..."
                font = cv2.FONT_HERSHEY_DUPLEX
                (text_w, text_h), _ = cv2.getTextSize(loading_text, font, 1.5, 3)
                text_x = (WINDOW_WIDTH - text_w) // 2
                text_y = WINDOW_HEIGHT // 2 - 50
                
                cv2.putText(loading_frame, loading_text, (text_x, text_y),
                           font, 1.5, (255, 255, 255), 3)
                
                # Sub text
                sub_text = "Loading MediaPipe models... Please wait (10-30 seconds)"
                (sub_w, sub_h), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                sub_x = (WINDOW_WIDTH - sub_w) // 2
                sub_y = text_y + 60
                
                cv2.putText(loading_frame, sub_text, (sub_x, sub_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
                
                cv2.imshow(WINDOW_NAME, loading_frame)
                cv2.waitKey(500)  # Show loading screen for 0.5 seconds
                
                # Close launcher window
                cv2.destroyAllWindows()
                
                # Launch the game
                should_continue = button_to_launch.launch_game(self.camera_index)
                
                if should_continue:
                    # Recreate launcher window
                    import time
                    time.sleep(0.2)  # Shorter delay for faster response
                    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
                    cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)
                    print("\n[LAUNCHER WINDOW RESTORED]\n")
                else:
                    self.running = False
                    break
            
            # Create frame
            frame = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
            
            # Draw all buttons
            for button in self.buttons:
                button.draw(frame)
            
            # Add title bar at the very top
            title_bar_height = 60
            title_bar = np.zeros((title_bar_height, WINDOW_WIDTH, 3), dtype=np.uint8)
            title_bar[:] = [30, 30, 30]
            
            title_text = "OPEN HOUSE - GAME LAUNCHER"
            title_font = cv2.FONT_HERSHEY_DUPLEX
            (title_w, title_h), _ = cv2.getTextSize(title_text, title_font, 1.2, 3)
            title_x = (WINDOW_WIDTH - title_w) // 2
            
            cv2.putText(title_bar, title_text,
                       (title_x, 40),
                       title_font, 1.2, (255, 255, 255), 3)
            
            # Overlay title bar
            frame[0:title_bar_height, :] = cv2.addWeighted(
                frame[0:title_bar_height, :], 0.3,
                title_bar, 0.7, 0
            )
            
            # Add instruction at bottom
            instruction = "Move mouse to preview | Click to launch game | Press 'Q' to quit"
            cv2.putText(frame, instruction,
                       (20, WINDOW_HEIGHT - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Display frame
            cv2.imshow(WINDOW_NAME, frame)
            
            # Handle keyboard
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == ord('Q'):  # Only Q quits (not ESC, games use ESC for back)
                self.running = False
                break
        
        cv2.destroyAllWindows()
        print("\n👋 Thanks for playing!")


def find_available_cameras():
    """Find all available camera indices"""
    available_cameras = []
    max_cameras_to_check = 5
    
    print("Scanning for available cameras...")
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
    
    return available_cameras


def choose_camera():
    """Let user choose from available cameras"""
    available_cameras = find_available_cameras()
    
    if not available_cameras:
        print("❌ No cameras found! Please check your camera connections.")
        exit()
    
    print("\n" + "="*60)
    print("📷 CAMERA SELECTION")
    print("="*60)
    
    for camera in available_cameras:
        print(f"  [{camera['index']}] Camera {camera['index']} - {camera['width']}x{camera['height']}")
    
    print("="*60)
    
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


def main():
    """Main entry point"""
    print("=" * 60)
    print("🎮 GAME LAUNCHER - OPEN HOUSE EDITION")
    print("=" * 60)
    print()
    
    # Select camera first
    camera_index = choose_camera()
    
    print(f"\n✅ Camera {camera_index} will be used for all games")
    print("\nLaunching game launcher...")
    print()
    
    # Launch the game launcher with selected camera
    launcher = GameLauncher(camera_index)
    launcher.run()


if __name__ == "__main__":
    main()
