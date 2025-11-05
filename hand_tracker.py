"""
Hand Tracker Module
Handles hand detection and gesture recognition using MediaPipe
"""

import cv2
import mediapipe as mp
import math


class HandTracker:
    """Tracks hands and recognizes gestures from webcam feed"""
    
    def __init__(self):
        """Initialize MediaPipe hands detector"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            max_num_hands=2,  # Track two hands for resize/rotate
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Gesture state
        self.current_gesture = None
        self.finger_positions = {}
        self.hand_detected = False
        self.landmarks = None
        self.frame_width = 0
        self.frame_height = 0
        
        # Two-hand tracking
        self.two_hands_detected = False
        self.left_hand_positions = {}
        self.right_hand_positions = {}
        self.left_landmarks = None
        self.right_landmarks = None
        
    def process_frame(self, frame):
        """
        Process frame and detect hand landmarks
        
        Args:
            frame: BGR image from camera
            
        Returns:
            results: MediaPipe hand landmarks results
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Store frame dimensions
        self.frame_height, self.frame_width = frame.shape[:2]
        
        if results.multi_hand_landmarks and results.multi_handedness:
            num_hands = len(results.multi_hand_landmarks)
            
            if num_hands == 1:
                # Single hand detected
                hand_landmarks = results.multi_hand_landmarks[0]
                handedness_label = results.multi_handedness[0].classification[0].label
                
                # Frame is already flipped before processing
                # MediaPipe "Right" = hand on right side of screen = user's right hand in mirror
                # MediaPipe "Left" = hand on left side of screen = user's left hand in mirror
                # We want only user's RIGHT hand for single-hand operations
                if handedness_label == "Right":  # User's right hand (right side of screen)
                    self.hand_detected = True
                    self.two_hands_detected = False
                    self.landmarks = hand_landmarks.landmark
                    self._extract_finger_positions(hand_landmarks, frame.shape)
                    self.current_gesture = self._recognize_gesture()
                else:
                    # Left hand alone cannot perform actions
                    self.hand_detected = False
                    self.two_hands_detected = False
                    self.current_gesture = None
                    self.finger_positions = {}
                    self.landmarks = None
            
            elif num_hands == 2:
                # Two hands detected
                self.hand_detected = True
                self.two_hands_detected = True
                
                # Identify left and right hands (frame is already flipped)
                for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                    label = handedness.classification[0].label
                    
                    # "Left" = left side of screen = user's left hand
                    # "Right" = right side of screen = user's right hand
                    if label == "Left":  # User's left hand
                        self.left_landmarks = hand_landmarks.landmark
                        self._extract_finger_positions_for_hand(hand_landmarks, frame.shape, 'left')
                    else:  # User's right hand
                        self.right_landmarks = hand_landmarks.landmark
                        self._extract_finger_positions_for_hand(hand_landmarks, frame.shape, 'right')
                
                # Use right hand (primary) for single-hand gestures
                if self.right_landmarks:
                    self.landmarks = self.right_landmarks
                    self.finger_positions = self.right_hand_positions
                
                # Check for two-hand gestures first
                two_hand_gesture = self._recognize_two_hand_gesture()
                if two_hand_gesture:
                    self.current_gesture = two_hand_gesture
                else:
                    # Only right hand can do single-hand gestures when both hands present
                    self.current_gesture = self._recognize_gesture()
        else:
            self.hand_detected = False
            self.two_hands_detected = False
            self.current_gesture = None
            self.finger_positions = {}
            self.landmarks = None
            self.left_hand_positions = {}
            self.right_hand_positions = {}
            self.left_landmarks = None
            self.right_landmarks = None
            
        return results
    
    def _extract_finger_positions(self, hand_landmarks, shape):
        """Extract key finger positions from landmarks"""
        h, w = shape[:2]
        
        # Key landmarks
        self.finger_positions = {
            'thumb_tip': (
                int(hand_landmarks.landmark[4].x * w),
                int(hand_landmarks.landmark[4].y * h)
            ),
            'index_tip': (
                int(hand_landmarks.landmark[8].x * w),
                int(hand_landmarks.landmark[8].y * h)
            ),
            'middle_tip': (
                int(hand_landmarks.landmark[12].x * w),
                int(hand_landmarks.landmark[12].y * h)
            ),
            'ring_tip': (
                int(hand_landmarks.landmark[16].x * w),
                int(hand_landmarks.landmark[16].y * h)
            ),
            'pinky_tip': (
                int(hand_landmarks.landmark[20].x * w),
                int(hand_landmarks.landmark[20].y * h)
            ),
            'index_mcp': (
                int(hand_landmarks.landmark[5].x * w),
                int(hand_landmarks.landmark[5].y * h)
            ),
            'middle_mcp': (
                int(hand_landmarks.landmark[9].x * w),
                int(hand_landmarks.landmark[9].y * h)
            ),
            'wrist': (
                int(hand_landmarks.landmark[0].x * w),
                int(hand_landmarks.landmark[0].y * h)
            )
        }
    
    def _extract_finger_positions_for_hand(self, hand_landmarks, shape, hand_type):
        """Extract finger positions for specific hand (left or right)"""
        h, w = shape[:2]
        
        positions = {
            'thumb_tip': (int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)),
            'index_tip': (int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)),
            'middle_tip': (int(hand_landmarks.landmark[12].x * w), int(hand_landmarks.landmark[12].y * h)),
            'ring_tip': (int(hand_landmarks.landmark[16].x * w), int(hand_landmarks.landmark[16].y * h)),
            'pinky_tip': (int(hand_landmarks.landmark[20].x * w), int(hand_landmarks.landmark[20].y * h)),
            'index_mcp': (int(hand_landmarks.landmark[5].x * w), int(hand_landmarks.landmark[5].y * h)),
            'middle_mcp': (int(hand_landmarks.landmark[9].x * w), int(hand_landmarks.landmark[9].y * h)),
            'wrist': (int(hand_landmarks.landmark[0].x * w), int(hand_landmarks.landmark[0].y * h))
        }
        
        if hand_type == 'left':
            self.left_hand_positions = positions
        else:
            self.right_hand_positions = positions
    
    def _recognize_two_hand_gesture(self):
        """Recognize gestures that require two hands"""
        if not self.left_hand_positions or not self.right_hand_positions:
            return None
        
        # Check for two-hand rotation: right hand pinch + left hand L-shape
        if self._is_left_hand_l_shape() and self._is_right_hand_pinch():
            return 'two_hand_rotate'
        
        # Check for two-hand resize: both hands pinching
        if self._is_left_hand_pinch() and self._is_right_hand_pinch():
            return 'two_hand_resize'
        
        return None
    
    def _is_left_hand_l_shape(self):
        """Check if left hand is in L-shape for rotation gesture"""
        if not self.left_hand_positions or not self.left_landmarks:
            return False
        
        # Get left hand positions
        thumb_tip = self.left_hand_positions.get('thumb_tip')
        index_tip = self.left_hand_positions.get('index_tip')
        middle_tip = self.left_hand_positions.get('middle_tip')
        wrist = self.left_hand_positions.get('wrist')
        
        if not thumb_tip or not index_tip or not middle_tip or not wrist:
            return False
        
        # Get left hand landmarks for more detailed checks
        h, w = self.frame_height, self.frame_width
        middle_pip_y = self.left_landmarks[10].y * h
        
        # Index should be up, middle should be down
        index_up = index_tip[1] < wrist[1] - 40
        middle_down = middle_tip[1] >= middle_pip_y - 10
        
        # Thumb should be extended horizontally
        thumb_extended = abs(thumb_tip[0] - wrist[0]) > 70
        
        # Thumb and index should be far apart (L-shape)
        thumb_index_distance = math.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
        
        return thumb_extended and index_up and middle_down and thumb_index_distance > 70
    
    def _is_left_hand_pinch(self):
        """Check if left hand is pinching (thumb and index close together)"""
        if not self.left_hand_positions:
            return False
        
        thumb_tip = self.left_hand_positions.get('thumb_tip')
        index_tip = self.left_hand_positions.get('index_tip')
        
        if not thumb_tip or not index_tip:
            return False
        
        distance = math.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
        return distance < 50
    
    def _is_right_hand_pinch(self):
        """Check if right hand is pinching (thumb and index close together)"""
        if not self.right_hand_positions:
            return False
        
        thumb_tip = self.right_hand_positions.get('thumb_tip')
        index_tip = self.right_hand_positions.get('index_tip')
        
        if not thumb_tip or not index_tip:
            return False
        
        distance = math.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
        return distance < 50
    
    def _recognize_gesture(self):
        """
        Recognize gesture based on finger positions
        
        Returns:
            str: Gesture name ('draw', 'pinch', 'palm', 'two_finger', 'three_finger', 'l_shape', None)
        """
        if not self.finger_positions:
            return None
        
        # Check for specific finger patterns first (more specific to less specific)
        
        # Check for L-shape (thumb + index extended, others down) - for rotation
        if self._is_l_shape():
            return 'l_shape'
        
        # Check for two fingers up (index + middle)
        if self._is_two_finger_gesture():
            return 'two_finger'
        
        # Check for three fingers up (index, middle, ring)
        if self._is_three_fingers_up():
            return 'three_finger'
        
        # Check for pinch (thumb and index close)
        if self._is_pinching():
            return 'pinch'
        
        # Index finger only (drawing) - check specifically for index up
        if self._is_index_only():
            return 'draw'
        
        # Count extended fingers for palm
        extended_fingers = self._count_extended_fingers()
        
        # Open palm (4-5 fingers)
        if extended_fingers >= 4:
            return 'palm'
        
        return None
    
    def _count_extended_fingers(self):
        """Count how many fingers are extended using PIP joints for accuracy"""
        if not self.finger_positions or not self.landmarks:
            return 0
        
        count = 0
        
        # Get PIP joint positions (joints between first and second segments)
        # Thumb IP: 3, Index PIP: 6, Middle PIP: 10, Ring PIP: 14, Pinky PIP: 18
        thumb_ip_y = self.landmarks[3].y * self.frame_height
        index_pip_y = self.landmarks[6].y * self.frame_height
        middle_pip_y = self.landmarks[10].y * self.frame_height
        ring_pip_y = self.landmarks[14].y * self.frame_height
        pinky_pip_y = self.landmarks[18].y * self.frame_height
        
        # Get finger tip positions
        thumb_tip = self.finger_positions['thumb_tip']
        index_tip = self.finger_positions['index_tip']
        middle_tip = self.finger_positions['middle_tip']
        ring_tip = self.finger_positions['ring_tip']
        pinky_tip = self.finger_positions['pinky_tip']
        wrist = self.finger_positions['wrist']
        
        # Thumb: check if tip is further out horizontally than IP joint
        thumb_ip_x = self.landmarks[3].x * self.frame_width
        thumb_extended = abs(thumb_tip[0] - wrist[0]) > abs(thumb_ip_x - wrist[0]) + 15
        
        # Other fingers: tip should be significantly above PIP joint (lower y = higher on screen)
        index_extended = index_tip[1] < index_pip_y - 20
        middle_extended = middle_tip[1] < middle_pip_y - 20
        ring_extended = ring_tip[1] < ring_pip_y - 20
        pinky_extended = pinky_tip[1] < pinky_pip_y - 20
        
        if thumb_extended:
            count += 1
        if index_extended:
            count += 1
        if middle_extended:
            count += 1
        if ring_extended:
            count += 1
        if pinky_extended:
            count += 1
            
        return count
    
    def _is_l_shape(self):
        """Check if hand is in L-shape (thumb and index extended, others down)"""
        if not self.finger_positions or not self.landmarks:
            return False
        
        # Get finger tips
        index_tip = self.finger_positions['index_tip']
        middle_tip = self.finger_positions['middle_tip']
        ring_tip = self.finger_positions['ring_tip']
        pinky_tip = self.finger_positions['pinky_tip']
        thumb_tip = self.finger_positions['thumb_tip']
        wrist = self.finger_positions['wrist']
        
        # Get PIP joints
        index_pip_y = self.landmarks[6].y * self.frame_height
        middle_pip_y = self.landmarks[10].y * self.frame_height
        ring_pip_y = self.landmarks[14].y * self.frame_height
        pinky_pip_y = self.landmarks[18].y * self.frame_height
        
        # Index finger must be extended
        index_up = index_tip[1] < index_pip_y - 30
        
        # Thumb must be extended (distance from wrist)
        thumb_extended = abs(thumb_tip[0] - wrist[0]) > 90
        
        # Other fingers must be down
        middle_down = middle_tip[1] >= middle_pip_y - 10
        ring_down = ring_tip[1] >= ring_pip_y - 10
        pinky_down = pinky_tip[1] >= pinky_pip_y - 10
        
        # Check if thumb and index are far apart (L-shape, not pointing)
        thumb_index_distance = math.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
        
        return index_up and thumb_extended and middle_down and ring_down and pinky_down and thumb_index_distance > 60
    
    def _is_index_only(self):
        """Check if ONLY the index finger is extended - works in ALL hand orientations (palm, back, side, pointing)"""
        if not self.finger_positions or not self.landmarks:
            return False
        
        import math
        
        # Get finger tips
        index_tip = self.finger_positions['index_tip']
        middle_tip = self.finger_positions['middle_tip']
        ring_tip = self.finger_positions['ring_tip']
        pinky_tip = self.finger_positions['pinky_tip']
        
        # Get MCP (knuckle) positions - base of each finger
        index_mcp = (int(self.landmarks[5].x * self.frame_width), int(self.landmarks[5].y * self.frame_height))
        middle_mcp = (int(self.landmarks[9].x * self.frame_width), int(self.landmarks[9].y * self.frame_height))
        ring_mcp = (int(self.landmarks[13].x * self.frame_width), int(self.landmarks[13].y * self.frame_height))
        pinky_mcp = (int(self.landmarks[17].x * self.frame_width), int(self.landmarks[17].y * self.frame_height))
        
        # Get PIP (middle joint) positions for better curl detection
        index_pip = (int(self.landmarks[6].x * self.frame_width), int(self.landmarks[6].y * self.frame_height))
        middle_pip = (int(self.landmarks[10].x * self.frame_width), int(self.landmarks[10].y * self.frame_height))
        ring_pip = (int(self.landmarks[14].x * self.frame_width), int(self.landmarks[14].y * self.frame_height))
        pinky_pip = (int(self.landmarks[18].x * self.frame_width), int(self.landmarks[18].y * self.frame_height))
        
        # Calculate tip-to-MCP distances (full finger length when extended)
        index_distance = math.sqrt((index_tip[0] - index_mcp[0])**2 + (index_tip[1] - index_mcp[1])**2)
        middle_distance = math.sqrt((middle_tip[0] - middle_mcp[0])**2 + (middle_tip[1] - middle_mcp[1])**2)
        ring_distance = math.sqrt((ring_tip[0] - ring_mcp[0])**2 + (ring_tip[1] - ring_mcp[1])**2)
        pinky_distance = math.sqrt((pinky_tip[0] - pinky_mcp[0])**2 + (pinky_tip[1] - pinky_mcp[1])**2)
        
        # Calculate tip-to-PIP distances (should be small if curled)
        middle_tip_to_pip = math.sqrt((middle_tip[0] - middle_pip[0])**2 + (middle_tip[1] - middle_pip[1])**2)
        ring_tip_to_pip = math.sqrt((ring_tip[0] - ring_pip[0])**2 + (ring_tip[1] - ring_pip[1])**2)
        pinky_tip_to_pip = math.sqrt((pinky_tip[0] - pinky_pip[0])**2 + (pinky_tip[1] - pinky_pip[1])**2)
        
        # Index finger must be extended (long distance from tip to knuckle)
        index_extended = index_distance > 85
        
        # Other fingers must be curled (two checks for accuracy):
        # 1. Tip-to-MCP distance is much shorter than index
        # 2. Tip-to-PIP distance is small (finger is bent at middle joint)
        middle_curled = (middle_distance < index_distance * 0.65) or (middle_tip_to_pip < 50)
        ring_curled = (ring_distance < index_distance * 0.65) or (ring_tip_to_pip < 50)
        pinky_curled = (pinky_distance < index_distance * 0.65) or (pinky_tip_to_pip < 45)
        
        # Check thumb is not extended (avoid confusion with pointing)
        thumb_tip = self.finger_positions['thumb_tip']
        thumb_mcp = (int(self.landmarks[2].x * self.frame_width), int(self.landmarks[2].y * self.frame_height))
        thumb_distance = math.sqrt((thumb_tip[0] - thumb_mcp[0])**2 + (thumb_tip[1] - thumb_mcp[1])**2)
        thumb_not_extended = thumb_distance < index_distance * 0.8
        
        return index_extended and middle_curled and ring_curled and pinky_curled and thumb_not_extended
    
    def _is_three_fingers_up(self):
        """Check if index, middle, and ring fingers are up"""
        if not self.finger_positions or not self.landmarks:
            return False
        
        # Get finger tips
        index_tip = self.finger_positions['index_tip']
        middle_tip = self.finger_positions['middle_tip']
        ring_tip = self.finger_positions['ring_tip']
        pinky_tip = self.finger_positions['pinky_tip']
        
        # Get PIP joints
        index_pip_y = self.landmarks[6].y * self.frame_height
        middle_pip_y = self.landmarks[10].y * self.frame_height
        ring_pip_y = self.landmarks[14].y * self.frame_height
        pinky_pip_y = self.landmarks[18].y * self.frame_height
        
        # Three fingers up
        index_up = index_tip[1] < index_pip_y - 25
        middle_up = middle_tip[1] < middle_pip_y - 25
        ring_up = ring_tip[1] < ring_pip_y - 25
        
        # Pinky down
        pinky_down = pinky_tip[1] >= pinky_pip_y - 10
        
        return index_up and middle_up and ring_up and pinky_down
    
    def _is_index_up(self):
        """Check if only index finger is up (legacy method)"""
        return self._is_index_only()
    
    def _is_pinching(self):
        """Check if thumb and index are pinching"""
        thumb = self.finger_positions['thumb_tip']
        index = self.finger_positions['index_tip']
        
        distance = math.sqrt((thumb[0] - index[0])**2 + (thumb[1] - index[1])**2)
        return distance < 40
    
    def _is_two_finger_gesture(self):
        """Check for two fingers up (index and middle extended, others down)"""
        if not self.finger_positions or not self.landmarks:
            return False
        
        # Get finger tips
        index_tip = self.finger_positions['index_tip']
        middle_tip = self.finger_positions['middle_tip']
        ring_tip = self.finger_positions['ring_tip']
        pinky_tip = self.finger_positions['pinky_tip']
        thumb_tip = self.finger_positions['thumb_tip']
        wrist = self.finger_positions['wrist']
        
        # Get PIP joints
        index_pip_y = self.landmarks[6].y * self.frame_height
        middle_pip_y = self.landmarks[10].y * self.frame_height
        ring_pip_y = self.landmarks[14].y * self.frame_height
        pinky_pip_y = self.landmarks[18].y * self.frame_height
        
        # Index and middle must be extended (tips well above PIP joints)
        index_up = index_tip[1] < index_pip_y - 30
        middle_up = middle_tip[1] < middle_pip_y - 30
        
        # Ring and pinky must be curled (tips at or below PIP joints)
        ring_down = ring_tip[1] >= ring_pip_y - 10
        pinky_down = pinky_tip[1] >= pinky_pip_y - 10
        
        # Thumb should be somewhat closed (not fully extended)
        thumb_closed = abs(thumb_tip[0] - wrist[0]) < 100
        
        return index_up and middle_up and ring_down and pinky_down
    
    def get_index_position(self):
        """Get index finger tip position"""
        if 'index_tip' in self.finger_positions:
            return self.finger_positions['index_tip']
        return None
    
    def get_pinch_center(self):
        """Get center point between thumb and index"""
        if 'thumb_tip' in self.finger_positions and 'index_tip' in self.finger_positions:
            thumb = self.finger_positions['thumb_tip']
            index = self.finger_positions['index_tip']
            return ((thumb[0] + index[0]) // 2, (thumb[1] + index[1]) // 2)
        return None
    
    def get_pinch_distance(self):
        """Get distance between thumb and index for resize gestures"""
        if 'thumb_tip' in self.finger_positions and 'index_tip' in self.finger_positions:
            thumb = self.finger_positions['thumb_tip']
            index = self.finger_positions['index_tip']
            distance = math.sqrt((thumb[0] - index[0])**2 + (thumb[1] - index[1])**2)
            return distance
        return None
    
    def get_hand_rotation(self):
        """Get hand rotation angle based on wrist to middle finger direction"""
        if 'wrist' in self.finger_positions and 'middle_mcp' in self.finger_positions:
            wrist = self.finger_positions['wrist']
            middle_mcp = self.finger_positions['middle_mcp']
            
            # Calculate angle from wrist to middle finger base
            dx = middle_mcp[0] - wrist[0]
            dy = middle_mcp[1] - wrist[1]
            angle = math.degrees(math.atan2(dy, dx))
            return angle
        return None
    
    def get_two_hand_distance(self):
        """Get distance between two pinch centers for resize (both hands pinching)"""
        if self.left_hand_positions and self.right_hand_positions:
            # Get pinch centers (midpoint between thumb and index for each hand)
            left_thumb = self.left_hand_positions.get('thumb_tip')
            left_index = self.left_hand_positions.get('index_tip')
            right_thumb = self.right_hand_positions.get('thumb_tip')
            right_index = self.right_hand_positions.get('index_tip')
            
            if left_thumb and left_index and right_thumb and right_index:
                left_pinch = ((left_thumb[0] + left_index[0]) // 2, (left_thumb[1] + left_index[1]) // 2)
                right_pinch = ((right_thumb[0] + right_index[0]) // 2, (right_thumb[1] + right_index[1]) // 2)
                
                distance = math.sqrt((left_pinch[0] - right_pinch[0])**2 + (left_pinch[1] - right_pinch[1])**2)
                return distance, left_pinch, right_pinch
        return None, None, None
    
    def get_two_hand_rotation_angle(self):
        """Get rotation angle for rotation (right hand pinch + left hand L-shape)"""
        if self.left_hand_positions and self.right_hand_positions:
            # Left hand index tip (L-shape pointer)
            left_index = self.left_hand_positions.get('index_tip')
            # Right hand pinch center
            right_thumb = self.right_hand_positions.get('thumb_tip')
            right_index = self.right_hand_positions.get('index_tip')
            
            if left_index and right_thumb and right_index:
                right_pinch = ((right_thumb[0] + right_index[0]) // 2, (right_thumb[1] + right_index[1]) // 2)
                
                # Calculate angle from right pinch to left index
                dx = left_index[0] - right_pinch[0]
                dy = left_index[1] - right_pinch[1]
                angle = math.degrees(math.atan2(-dy, dx))  # Negative dy for screen coordinates
                return angle, left_index, right_pinch
        return None, None, None
    
    def get_two_hand_center(self):
        """Get center point between two hands for selection"""
        if self.left_hand_positions and self.right_hand_positions:
            # For resize: center between two pinches
            left_thumb = self.left_hand_positions.get('thumb_tip')
            left_index = self.left_hand_positions.get('index_tip')
            right_thumb = self.right_hand_positions.get('thumb_tip')
            right_index = self.right_hand_positions.get('index_tip')
            
            if left_thumb and left_index and right_thumb and right_index:
                left_pinch = ((left_thumb[0] + left_index[0]) // 2, (left_thumb[1] + left_index[1]) // 2)
                right_pinch = ((right_thumb[0] + right_index[0]) // 2, (right_thumb[1] + right_index[1]) // 2)
                center = ((left_pinch[0] + right_pinch[0]) // 2, (left_pinch[1] + right_pinch[1]) // 2)
                return center
        return None
    
    def draw_landmarks(self, frame, results):
        """Draw hand landmarks on frame"""
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                )
    
    def release(self):
        """Release resources"""
        self.hands.close()
