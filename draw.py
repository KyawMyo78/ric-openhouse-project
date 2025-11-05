"""
Virtual Air Drawing Application
Main entry point for hand gesture-controlled 2D drawing application

Features:
- Webcam-based hand tracking
- Draw in air with gestures
- 2D shapes (lines, circles, rectangles, triangles)
- Gesture controls for drawing, selecting, moving, resizing
- Interactive UI with color palette and shape menu

Gesture Controls:
- Index finger up: Draw mode
- Pinch (thumb + index): Select/Move objects
- Three fingers up: Show shape menu
- Open palm: Stop drawing

Keyboard Controls:
- 'C': Clear canvas
- 'H': Toggle help
- 'U': Undo last action
- 'Q': Quit application
"""

import cv2
import numpy as np
import time
from hand_tracker import HandTracker
from draw_manager import DrawManager
from ui_manager import UIManager


class VirtualDrawingApp:
    """Main application class"""
    
    def __init__(self):
        """Initialize application"""
        print("=" * 60)
        print("Virtual Air Drawing Application")
        print("=" * 60)
        print("\n[INITIALIZING...]")
        
        # Video capture
        self.cap = None
        self.width = 1280
        self.height = 720
        
        # Initialize modules
        self.hand_tracker = HandTracker()
        self.draw_manager = DrawManager(self.width, self.height)
        self.ui_manager = UIManager(self.width, self.height)
        
        # State
        self.last_gesture = None
        self.gesture_start_time = 0
        self.gesture_cooldown = 0.5  # seconds
        self.last_valid_gesture_time = 0  # For gesture stability
        self.gesture_timeout = 0.5  # seconds - how long to wait before stopping action
        
        # For smooth transitions
        self.last_draw_position = None
        self.is_drawing = False
        self.is_moving = False
        self.is_resizing = False
        self.is_rotating = False
        
        # Mouse support
        self.mouse_position = None
        
        # Button selection cooldown
        self.last_button_press = 0
        self.button_cooldown = 1.0  # seconds
        
        # For resize/rotate gestures
        self.initial_pinch_distance = None
        self.initial_rotation = None
        self.operation_mode = 'move'  # 'move', 'resize', 'rotate'
        
        print("[OK] Modules initialized")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for UI interactions"""
        self.mouse_position = (x, y)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicked on color palette
            if self.ui_manager.color_palette.visible:
                for i, color in enumerate(self.ui_manager.color_palette.colors):
                    px = self.ui_manager.color_palette.x + i * (self.ui_manager.color_palette.button_size + 5)
                    py = self.ui_manager.color_palette.y
                    size = self.ui_manager.color_palette.button_size
                    
                    if px <= x <= px + size and py <= y <= py + size:
                        self.ui_manager.color_palette.selected_index = i
                        self.draw_manager.current_color = color
                        print(f"[COLOR SELECTED] Color {i}: {color}")
                        return
            
            # Check if clicked on shape menu
            if self.ui_manager.shape_menu.visible:
                for i, (shape_name, button) in enumerate(self.ui_manager.shape_menu.buttons.items()):
                    if button.contains((x, y)):
                        self.ui_manager.shape_menu.selected_shape = shape_name
                        # Set all buttons inactive
                        for btn in self.ui_manager.shape_menu.buttons.values():
                            btn.active = False
                        # Set clicked button active
                        button.active = True
                        print(f"[SHAPE SELECTED] {shape_name}")
                        return
    
    def find_available_cameras(self):
        """Find all available camera indices"""
        available_cameras = []
        max_cameras_to_check = 5
        
        print("\n[SCANNING FOR CAMERAS]")
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
    
    def choose_camera(self):
        """Let user choose from available cameras"""
        available_cameras = self.find_available_cameras()
        
        if not available_cameras:
            print("\n❌ No cameras found! Please check your camera connections.")
            return None
        
        print("\n" + "="*60)
        print("📷 AVAILABLE CAMERAS:")
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
                return None
        
    def setup_camera(self):
        """Setup camera capture"""
        print("\n[CAMERA SETUP]")
        
        # Let user choose camera
        camera_index = self.choose_camera()
        
        if camera_index is None:
            return False
        
        # Open selected camera
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        
        if not self.cap or not self.cap.isOpened():
            print(f"✗ Failed to open camera {camera_index}!")
            return False
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        print(f"\n✓ Camera {camera_index} opened successfully")
        print(f"✓ Resolution: {actual_width}x{actual_height} @ {actual_fps}fps")
        
        # Warm up camera
        print("⚡ Warming up camera...")
        for _ in range(10):
            self.cap.read()
        
        print("✓ Camera ready!")
        return True
    
    def process_gesture(self, gesture):
        """
        Process recognized gesture and perform action with stability
        
        Args:
            gesture: Gesture name from HandTracker
        """
        current_time = time.time()
        
        # Update UI feedback
        gesture_names = {
            'draw': 'Drawing',
            'pinch': 'Select/Move',
            'two_hand_resize': 'Resizing (2 hands)',
            'two_hand_rotate': 'Rotating (2 hands)',
            'palm': 'Stopped',
            'three_finger': 'Menu'
        }
        
        # If we have a valid gesture, update timestamp
        if gesture:
            self.last_gesture = gesture
            self.last_valid_gesture_time = current_time
            self.ui_manager.set_gesture_text(gesture_names.get(gesture, ''))
        else:
            # No gesture detected - check if we should continue with last gesture (stability)
            time_since_last_gesture = current_time - self.last_valid_gesture_time
            if time_since_last_gesture <= self.gesture_timeout:
                # Within stability timeout - continue with last gesture
                gesture = self.last_gesture
                if gesture:
                    self.ui_manager.set_gesture_text(gesture_names.get(gesture, '') + " (hold)")
            else:
                # Timeout exceeded - stop actions
                if self.is_drawing or self.is_moving or self.is_resizing or self.is_rotating:
                    self.stop_actions()
                self.ui_manager.set_gesture_text("")
                return
        
        # Handle gestures (either current or continued from last)
        if gesture == 'draw':
            self.handle_draw_gesture()
        elif gesture == 'pinch':
            self.handle_pinch_gesture()
        elif gesture == 'two_hand_resize':
            self.handle_two_hand_resize()
        elif gesture == 'two_hand_rotate':
            self.handle_two_hand_rotate()
        elif gesture == 'palm':
            self.handle_palm_gesture()
        elif gesture == 'three_finger':
            self.handle_three_finger_gesture()
    
    def handle_draw_gesture(self):
        """Handle drawing gesture"""
        position = self.hand_tracker.get_index_position()
        if not position:
            return
        
        # 2D drawing
        if not self.is_drawing:
            self.draw_manager.start_drawing(position)
            self.is_drawing = True
        else:
            self.draw_manager.update_drawing(position)
        
        self.last_draw_position = position
    
    def handle_pinch_gesture(self):
        """Handle pinch gesture (select/move only)"""
        position = self.hand_tracker.get_pinch_center()
        
        if not position:
            return
        
        if not self.is_moving:
            # Start move - select object first
            selected = self.draw_manager.select_object(position)
            if selected:
                self.draw_manager.start_move(position)
                self.is_moving = True
        else:
            # Continue moving
            self.draw_manager.update_move(position)
        
        self.last_draw_position = position
    
    def handle_two_hand_resize(self):
        """Handle two-hand resize gesture (both hands pinching)"""
        distance, left_pinch, right_pinch = self.hand_tracker.get_two_hand_distance()
        center = self.hand_tracker.get_two_hand_center()
        
        if not distance or not center:
            return
        
        if not self.is_resizing:
            # Start resize - select object at center
            selected = self.draw_manager.select_object(center)
            if selected:
                self.initial_pinch_distance = distance
                self.is_resizing = True
                # Store initial size
                if self.draw_manager.selected_object:
                    self.draw_manager.selected_object.initial_size = self.draw_manager.selected_object.size
                    if hasattr(self.draw_manager.selected_object, 'width'):
                        self.draw_manager.selected_object.initial_width = self.draw_manager.selected_object.width
                        self.draw_manager.selected_object.initial_height = self.draw_manager.selected_object.height
                print(f"[RESIZE] Started - Initial distance: {distance:.1f}px")
        else:
            # Continue resize
            if self.draw_manager.selected_object and self.initial_pinch_distance:
                scale = distance / self.initial_pinch_distance
                scale = max(0.3, min(scale, 5.0))
                obj_type = self.draw_manager.selected_object.shape_type
                print(f"[RESIZE] {obj_type} - Distance: {distance:.1f}px, Scale: {scale:.2f}x")
                self.draw_manager.selected_object.resize(scale)
    
    def handle_two_hand_rotate(self):
        """Handle two-hand rotation gesture (right hand pinch selects + left hand L-shape rotates)"""
        angle, left_index, right_pinch = self.hand_tracker.get_two_hand_rotation_angle()
        
        if angle is None or not right_pinch:
            return
        
        if not self.is_rotating:
            # Start rotation - select object at right hand pinch position
            selected = self.draw_manager.select_object(right_pinch)
            if selected:
                self.initial_rotation = angle
                self.is_rotating = True
                print(f"[ROTATION] Started - Initial angle: {angle:.1f}°")
        else:
            # Continue rotation
            if self.draw_manager.selected_object and self.initial_rotation is not None:
                rotation_delta = angle - self.initial_rotation
                self.draw_manager.selected_object.rotation = rotation_delta
                print(f"[ROTATION] Angle: {angle:.1f}°, Delta: {rotation_delta:.1f}°")
    
    def handle_palm_gesture(self):
        """Handle palm gesture (stop)"""
        self.stop_actions()
    
    def handle_three_finger_gesture(self):
        """Handle three-finger gesture (show menu)"""
        self.ui_manager.shape_menu.visible = True
        self.ui_manager.color_palette.visible = True
    
    def check_ui_selection(self):
        """Check if finger is pointing at UI elements for selection"""
        position = self.hand_tracker.get_index_position()
        if not position:
            return
        
        x, y = position
        current_time = time.time()
        
        # Check clear button (with cooldown to prevent multiple triggers)
        clear_btn = self.ui_manager.clear_button
        if clear_btn.visible and clear_btn.contains((x, y)):
            if current_time - self.last_button_press > self.button_cooldown:
                self.last_button_press = current_time
                self.draw_manager.clear_canvas()
                print("[CLEAR] Canvas cleared")
                return
        
        # Check color palette selection
        if self.ui_manager.color_palette.visible:
            for i, color in enumerate(self.ui_manager.color_palette.colors):
                px = self.ui_manager.color_palette.x + i * (self.ui_manager.color_palette.button_size + 5)
                py = self.ui_manager.color_palette.y
                size = self.ui_manager.color_palette.button_size
                
                if px <= x <= px + size and py <= y <= py + size:
                    if self.ui_manager.color_palette.selected_index != i:
                        self.ui_manager.color_palette.selected_index = i
                        self.draw_manager.current_color = color
                        print(f"[COLOR SELECTED] Color {i}: {color}")
                    return
        
        # Check shape menu selection
        if self.ui_manager.shape_menu.visible:
            shapes_list = self.ui_manager.shape_menu.shapes
            for i, shape_name in enumerate(shapes_list):
                sx = self.ui_manager.shape_menu.x
                sy = self.ui_manager.shape_menu.y + i * (self.ui_manager.shape_menu.button_size + 10)
                size = self.ui_manager.shape_menu.button_size
                
                if sx <= x <= sx + size and sy <= y <= sy + size:
                    if self.ui_manager.shape_menu.selected_shape != shape_name:
                        self.ui_manager.shape_menu.selected_shape = shape_name
                        self.draw_manager.current_shape = shape_name  # Update draw manager
                        # Update button states
                        for btn_name, btn in self.ui_manager.shape_menu.buttons.items():
                            btn.active = (btn_name == shape_name)
                        print(f"[SHAPE SELECTED] {shape_name}")
                    return
    
    def draw_pinch_feedback(self, frame):
        """Draw visual feedback for pinch move gesture"""
        thumb_tip = self.hand_tracker.finger_positions.get('thumb_tip')
        index_tip = self.hand_tracker.finger_positions.get('index_tip')
        
        if thumb_tip and index_tip:
            # Draw line between thumb and index
            cv2.line(frame, thumb_tip, index_tip, (0, 255, 255), 3)
            
            # Draw circles at fingertips
            cv2.circle(frame, thumb_tip, 8, (0, 255, 255), -1)
            cv2.circle(frame, index_tip, 8, (0, 255, 255), -1)
            
            # Display "Moving" text
            mid_x = (thumb_tip[0] + index_tip[0]) // 2
            mid_y = (thumb_tip[1] + index_tip[1]) // 2 - 20
            
            text = "Moving"
            
            # Draw background for text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (mid_x - text_size[0]//2 - 5, mid_y - text_size[1] - 5),
                         (mid_x + text_size[0]//2 + 5, mid_y + 5), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(frame, text, (mid_x - text_size[0]//2, mid_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def draw_rotation_feedback(self, frame):
        """Draw visual feedback for rotation gesture"""
        wrist = self.hand_tracker.finger_positions.get('wrist')
        middle_mcp = self.hand_tracker.finger_positions.get('middle_mcp')
        rotation_angle = self.hand_tracker.get_hand_rotation()
        
        if wrist and middle_mcp and rotation_angle is not None:
            # Draw rotation arc
            center = wrist
            radius = 80
            
            # Draw base circle
            cv2.circle(frame, center, radius, (255, 100, 255), 2)
            
            # Draw angle line
            import numpy as np
            end_x = int(center[0] + radius * np.cos(np.radians(rotation_angle)))
            end_y = int(center[1] + radius * np.sin(np.radians(rotation_angle)))
            cv2.line(frame, center, (end_x, end_y), (255, 100, 255), 3)
            
            # Draw arc from initial to current angle
            if self.initial_rotation is not None:
                start_angle = int(self.initial_rotation)
                end_angle = int(rotation_angle)
                cv2.ellipse(frame, center, (radius, radius), 0, start_angle, end_angle, (255, 0, 255), 3)
                
                rotation_delta = rotation_angle - self.initial_rotation
                text = f"Rotation: {int(rotation_delta)}°"
            else:
                text = f"Angle: {int(rotation_angle)}°"
            
            # Draw text
            text_pos = (center[0] - 60, center[1] - radius - 15)
            cv2.rectangle(frame, (text_pos[0] - 5, text_pos[1] - 25),
                         (text_pos[0] + 140, text_pos[1] + 5), (0, 0, 0), -1)
            cv2.putText(frame, text, text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 255), 2)
    
    def draw_two_hand_resize_feedback(self, frame):
        """Draw visual feedback for two-hand resize gesture (both hands pinching)"""
        distance, left_pinch, right_pinch = self.hand_tracker.get_two_hand_distance()
        
        if distance and left_pinch and right_pinch:
            # Draw line between two pinch centers
            cv2.line(frame, left_pinch, right_pinch, (0, 255, 0), 4)
            
            # Draw circles at pinch centers
            cv2.circle(frame, left_pinch, 15, (0, 255, 0), -1)
            cv2.circle(frame, right_pinch, 15, (0, 255, 0), -1)
            
            # Draw pinch indicators for both hands
            if self.hand_tracker.left_hand_positions:
                left_thumb = self.hand_tracker.left_hand_positions.get('thumb_tip')
                left_index = self.hand_tracker.left_hand_positions.get('index_tip')
                if left_thumb and left_index:
                    cv2.line(frame, left_thumb, left_index, (0, 255, 0), 2)
                    cv2.circle(frame, left_thumb, 6, (0, 255, 0), -1)
                    cv2.circle(frame, left_index, 6, (0, 255, 0), -1)
            
            if self.hand_tracker.right_hand_positions:
                right_thumb = self.hand_tracker.right_hand_positions.get('thumb_tip')
                right_index = self.hand_tracker.right_hand_positions.get('index_tip')
                if right_thumb and right_index:
                    cv2.line(frame, right_thumb, right_index, (0, 255, 0), 2)
                    cv2.circle(frame, right_thumb, 6, (0, 255, 0), -1)
                    cv2.circle(frame, right_index, 6, (0, 255, 0), -1)
            
            # Draw distance and scale text
            mid_point = ((left_pinch[0] + right_pinch[0]) // 2, (left_pinch[1] + right_pinch[1]) // 2 - 25)
            
            if self.is_resizing and self.initial_pinch_distance:
                scale = distance / self.initial_pinch_distance
                text = f"Resize: {scale:.2f}x ({int(distance)}px)"
            else:
                text = f"Distance: {int(distance)}px"
            
            # Draw background for text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (mid_point[0] - text_size[0]//2 - 5, mid_point[1] - text_size[1] - 5),
                         (mid_point[0] + text_size[0]//2 + 5, mid_point[1] + 5), (0, 0, 0), -1)
            
            cv2.putText(frame, text, (mid_point[0] - text_size[0]//2, mid_point[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def draw_two_hand_rotate_feedback(self, frame):
        """Draw visual feedback for two-hand rotation gesture (right pinch + left L-shape)"""
        angle, left_index, right_pinch = self.hand_tracker.get_two_hand_rotation_angle()
        
        if angle is not None and left_index and right_pinch:
            # Draw line from right pinch to left index
            cv2.line(frame, right_pinch, left_index, (255, 0, 255), 4)
            
            # Draw right hand pinch
            if self.hand_tracker.right_hand_positions:
                right_thumb = self.hand_tracker.right_hand_positions.get('thumb_tip')
                right_index_tip = self.hand_tracker.right_hand_positions.get('index_tip')
                if right_thumb and right_index_tip:
                    cv2.line(frame, right_thumb, right_index_tip, (255, 0, 255), 2)
                    cv2.circle(frame, right_thumb, 6, (255, 0, 255), -1)
                    cv2.circle(frame, right_index_tip, 6, (255, 0, 255), -1)
            
            # Draw left hand L-shape
            if self.hand_tracker.left_hand_positions:
                left_thumb = self.hand_tracker.left_hand_positions.get('thumb_tip')
                left_wrist = self.hand_tracker.left_hand_positions.get('wrist')
                if left_thumb and left_index and left_wrist:
                    # Draw L-shape
                    cv2.line(frame, left_wrist, left_thumb, (255, 0, 255), 3)
                    cv2.line(frame, left_wrist, left_index, (255, 0, 255), 3)
                    cv2.circle(frame, left_thumb, 8, (255, 0, 255), -1)
                    cv2.circle(frame, left_index, 8, (255, 0, 255), -1)
            
            # Draw pinch center and index tip
            cv2.circle(frame, right_pinch, 15, (255, 0, 255), -1)
            cv2.circle(frame, left_index, 12, (255, 100, 255), -1)
            
            # Draw rotation circle at right pinch (pivot point)
            radius = 70
            cv2.circle(frame, right_pinch, radius, (255, 100, 255), 2)
            
            # Draw angle line
            import numpy as np
            end_x = int(right_pinch[0] + radius * np.cos(np.radians(angle)))
            end_y = int(right_pinch[1] - radius * np.sin(np.radians(angle)))
            cv2.line(frame, right_pinch, (end_x, end_y), (255, 0, 255), 3)
            
            # Draw rotation arc if rotating
            if self.initial_rotation is not None:
                start_angle = int(-self.initial_rotation)
                end_angle = int(-angle)
                cv2.ellipse(frame, right_pinch, (radius, radius), 0, start_angle, end_angle, (255, 50, 255), 3)
                
                rotation_delta = angle - self.initial_rotation
                text = f"Rotate: {int(rotation_delta)}°"
            else:
                text = f"Angle: {int(angle)}°"
            
            # Draw text
            text_pos = (right_pinch[0] - 70, right_pinch[1] - radius - 20)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (text_pos[0] - 5, text_pos[1] - text_size[1] - 5),
                         (text_pos[0] + text_size[0] + 5, text_pos[1] + 5), (0, 0, 0), -1)
            cv2.putText(frame, text, text_pos,
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    def stop_actions(self):
        """Stop all ongoing actions"""
        if self.is_drawing:
            self.draw_manager.end_drawing()
            self.is_drawing = False
        
        if self.is_moving:
            self.draw_manager.end_move()
            self.is_moving = False
        
        if self.is_resizing:
            self.is_resizing = False
        
        if self.is_rotating:
            self.is_rotating = False
        
        # Reset operation state
        self.last_draw_position = None
        self.initial_pinch_distance = None
        self.initial_rotation = None
        self.operation_mode = 'move'
        
        # Keep menus visible - don't hide them
        # self.ui_manager.shape_menu.visible = False
        # self.ui_manager.color_palette.visible = False
    
    def handle_keyboard(self, key):
        """
        Handle keyboard input
        
        Args:
            key: Pressed key code
        """
        if key == ord('q') or key == ord('Q'):
            return False  # Quit
        elif key == ord('c') or key == ord('C'):
            # Clear canvas
            self.draw_manager.clear_canvas()
            print("[CLEARED] Canvas cleared")
        elif key == ord('h') or key == ord('H'):
            # Toggle help
            self.ui_manager.toggle_help()
        elif key == ord('u') or key == ord('U'):
            # Undo
            self.draw_manager.undo()
            print("[UNDO] Last action undone")
        elif key == ord('s') or key == ord('S'):
            # Toggle shape menu
            self.ui_manager.shape_menu.visible = not self.ui_manager.shape_menu.visible
            print(f"[MENU] Shape menu {'shown' if self.ui_manager.shape_menu.visible else 'hidden'}")
        elif key == ord('p') or key == ord('P'):
            # Toggle color palette
            self.ui_manager.color_palette.visible = not self.ui_manager.color_palette.visible
            print(f"[PALETTE] Color palette {'shown' if self.ui_manager.color_palette.visible else 'hidden'}")
        
        return True
    
    def run(self):
        """Main application loop"""
        if not self.setup_camera():
            print("Failed to setup camera. Exiting.")
            return
        
        print("\n" + "=" * 60)
        print("APPLICATION STARTED")
        print("=" * 60)
        print("\n[CONTROLS]")
        print("Gestures:")
        print("  - 1 Finger Up: Draw mode")
        print("  - Pinch (thumb+index close): Select object")
        print("    • Keep pinch & move: Move object")
        print("    • Keep pinch & spread/squeeze: Resize object (shows distance)")
        print("  - L-Shape (thumb+index extended): Rotate object (shows angle)")
        print("  - 2 Fingers Up (peace sign): Toggle 2D/3D mode")
        print("  - 3 Fingers Up: Show menus")
        print("  - Open Palm: Stop action")
        print("\nKeyboard:")
        print("  - 'M': Toggle 2D/3D mode")
        print("  - 'S': Toggle shape menu")
        print("  - 'P': Toggle color palette")
        print("  - 'C': Clear canvas")
        print("  - 'H': Toggle help overlay")
        print("  - 'U': Undo last action")
        print("  - 'Q': Quit")
        print("\nFinger Selection:")
        print("  - Point at colors (top) to select")
        print("  - Point at shapes (right) to select")
        print("  - Point at Clear button to clear canvas")
        print("\n" + "=" * 60)
        
        # Create window
        cv2.namedWindow('Virtual Air Drawing', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Virtual Air Drawing', self.width, self.height)
        cv2.setMouseCallback('Virtual Air Drawing', self.mouse_callback)
        
        # FPS calculation
        fps_start_time = time.time()
        fps_counter = 0
        fps_display = 0
        
        # Main loop
        running = True
        while running:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (self.width, self.height))
            
            # Process hand tracking
            results = self.hand_tracker.process_frame(frame)
            
            # Draw hand landmarks
            self.hand_tracker.draw_landmarks(frame, results)
            
            # Process gestures
            if self.hand_tracker.hand_detected:
                gesture = self.hand_tracker.current_gesture
                # Always process gesture (even if None) for stability handling
                self.process_gesture(gesture)
                
                # Check for UI selection with index finger (only when not moving objects)
                if not self.is_moving and gesture != 'pinch':
                    self.check_ui_selection()
                
                # Draw visual feedback for pinch (move only now)
                if gesture == 'pinch' and self.is_moving:
                    self.draw_pinch_feedback(frame)
                
                # Draw visual feedback for two-hand resize
                if gesture == 'two_hand_resize' and self.is_resizing:
                    self.draw_two_hand_resize_feedback(frame)
                
                # Draw visual feedback for two-hand rotation
                if gesture == 'two_hand_rotate' and self.is_rotating:
                    self.draw_two_hand_rotate_feedback(frame)
            else:
                # No hand detected - use gesture stability timeout
                current_time = time.time()
                time_since_last_gesture = current_time - self.last_valid_gesture_time
                if time_since_last_gesture > self.gesture_timeout:
                    self.stop_actions()
                    self.ui_manager.set_gesture_text("No hand detected")
                else:
                    # Keep showing last gesture briefly
                    if self.last_gesture:
                        gesture_names = {
                            'draw': 'Drawing',
                            'pinch': 'Select/Move',
                            'two_hand_resize': 'Resizing',
                            'two_hand_rotate': 'Rotating'
                        }
                        self.ui_manager.set_gesture_text(gesture_names.get(self.last_gesture, '') + " (searching hand...)")
            
            # Render drawings
            self.draw_manager.draw_all(frame)
            
            # Draw UI
            self.ui_manager.draw(frame)
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_start_time > 1.0:
                fps_display = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            # Draw FPS
            cv2.putText(frame, f"FPS: {fps_display}", (self.width - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Virtual Air Drawing', frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if key != 255:
                if not self.handle_keyboard(key):
                    running = False
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\n[CLEANUP]")
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.hand_tracker.release()
        print("✓ Resources released")
        print("\nThank you for using Virtual Air Drawing!")


def main():
    """Main entry point"""
    try:
        app = VirtualDrawingApp()
        app.run()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Exiting...")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
