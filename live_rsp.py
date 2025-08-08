import cv2
import mediapipe as mp
import random
import time
import platform

# Sound functions
if platform.system() == 'Windows':
    import winsound
    def beep(): winsound.Beep(1000, 200)
else:
    import os
    def beep(): os.system('play -nq -t alsa synth 0.2 sine 1000')  # or use 'afplay' on Mac

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.7)

def test_camera_resolution(camera_index, width, height):
    """Test if a camera supports a specific resolution"""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        return False, None
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Read a frame to verify the resolution works
    ret, frame = cap.read()
    if ret:
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return True, (actual_width, actual_height)
    
    cap.release()
    return False, None

def find_available_cameras():
    """Find all available camera indices"""
    available_cameras = []
    max_cameras_to_check = 5  # Check cameras 0-4
    
    print("Scanning for available cameras...")
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            # Try to read a frame to verify the camera works
            ret, frame = cap.read()
            if ret:
                # Get camera info
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
    """Let user choose from available cameras"""
    available_cameras = find_available_cameras()
    
    if not available_cameras:
        print("❌ No cameras found! Please check your camera connections.")
        exit()
    
    print("\n" + "="*50)
    print("📷 AVAILABLE CAMERAS:")
    print("="*50)
    
    for i, camera in enumerate(available_cameras):
        print(f"  [{camera['index']}] Camera {camera['index']} - {camera['width']}x{camera['height']}")
    
    print("="*50)
    
    while True:
        try:
            user_input = input(f"Choose camera index (available: {[cam['index'] for cam in available_cameras]}): ").strip()
            selected_index = int(user_input)
            
            # Check if the selected camera is available
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

def find_best_camera_resolution(camera_index):
    """Find the best available resolution for a camera"""
    # Common resolutions to test (from highest to lowest)
    resolutions = [
        (1920, 1080),  # Full HD
        (1280, 720),   # HD
        (960, 720),    # HD ready
        (800, 600),    # SVGA
        (640, 480),    # VGA
        (320, 240)     # QVGA
    ]
    
    print(f"Testing resolutions for camera {camera_index}...")
    for width, height in resolutions:
        success, actual_res = test_camera_resolution(camera_index, width, height)
        if success and actual_res:
            print(f"  ✓ {width}x{height} -> Actually got: {actual_res[0]}x{actual_res[1]}")
            return actual_res
        else:
            print(f"  ✗ {width}x{height} - Not supported")
    
    # If all fail, return default
    print(f"  Using default resolution for camera {camera_index}")
    return (640, 480)

print("🎮 Rock Paper Scissors with Emotion Detection")
print("=" * 50)

# Let user choose camera
camera_index = choose_camera()

print(f"\n📷 Opening camera {camera_index}...")
start_time = time.time()

# Open selected camera
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

if not cap.isOpened():
    print(f"❌ Error: Could not open camera {camera_index}")
    exit()

print(f"✅ Camera {camera_index} opened successfully!")
best_resolution = find_best_camera_resolution(camera_index)
cap.release()  # Release to reopen with best settings

# Reopen with best resolution
cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, best_resolution[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best_resolution[1])

print(f"⏱️  Camera {camera_index} opened in {time.time() - start_time:.2f} seconds")
print(f"🎯 Using resolution: {best_resolution[0]}x{best_resolution[1]}")

# Warm up camera with dummy reads
print("🔥 Warming up camera...")
for i in range(5):
    ret, frame = cap.read()
    if ret:
        print(f"  📸 Warm-up frame {i+1}/5 captured")
    else:
        print(f"  ⚠️  Warning: Failed to capture warm-up frame {i+1}")

print("✅ Camera ready!")

# Display actual camera resolution
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"📐 Final camera resolution: {actual_width}x{actual_height}")

# Auto-adjust display window size and UI scaling based on camera resolution
if actual_width >= 1280:
    display_width = 1200  # Large window for high-res cameras
    ui_scale = 1.0        # Full scale UI
    text_scale = 1.0      # Full scale text
    overlay_height = 250  # Standard overlay height
elif actual_width >= 800:
    display_width = 900   # Medium window for medium-res cameras
    ui_scale = 0.8        # Slightly smaller UI
    text_scale = 0.8      # Slightly smaller text
    overlay_height = 200  # Reduced overlay height
else:
    display_width = 700   # Smaller window for low-res cameras
    ui_scale = 0.6        # Much smaller UI for low-res
    text_scale = 0.6      # Much smaller text
    overlay_height = 150  # Much smaller overlay height

print(f"🖥️  Display window width will be: {display_width}px")
print(f"📏 UI scaling factor: {ui_scale}x")
print(f"📝 Text scaling factor: {text_scale}x")
print("=" * 50)

gesture_emojis = {
    "rock": "ROCK",
    "paper": "PAPER",
    "scissors": "SCISSORS"
}

# Game state
result_text = ""
player_move = ""
computer_move = ""
player_score = 0
computer_score = 0
round_active = True
countdown_started = False
countdown_start_time = 0
current_emotion = "neutral"
emotion_confidence = 0.0
show_landmarks = False  # Toggle for showing landmarks
emotion_emojis = {
    "happy": "HAPPY",
    "sad": "SAD", 
    "surprised": "SURPRISED",
    "sleepy": "SLEEPY",
    "neutral": "NEUTRAL"
}
countdown_duration = 3

def detect_emotion(face_landmarks):
    """
    Improved emotion detection based on facial landmarks
    Returns: emotion string and confidence
    """
    if not face_landmarks:
        return "neutral", 0.0
    
    landmarks = face_landmarks.landmark
    
    # More accurate landmark indices for emotion detection
    # Mouth landmarks
    mouth_left = landmarks[61]    # Left mouth corner
    mouth_right = landmarks[291]  # Right mouth corner
    mouth_top = landmarks[13]     # Upper lip top
    mouth_bottom = landmarks[14]  # Lower lip bottom
    upper_lip = landmarks[12]     # Upper lip center
    lower_lip = landmarks[15]     # Lower lip center
    
    # Eye landmarks for better surprise detection
    left_eye_top = landmarks[159]
    left_eye_bottom = landmarks[145]
    right_eye_top = landmarks[386]
    right_eye_bottom = landmarks[374]
    
    # Additional eye landmarks for width detection
    left_eye_inner = landmarks[133]
    left_eye_outer = landmarks[33]
    right_eye_inner = landmarks[362]
    right_eye_outer = landmarks[263]
    
    # Eyebrow landmarks for surprise detection
    left_eyebrow_inner = landmarks[70]
    left_eyebrow_outer = landmarks[107]
    right_eyebrow_inner = landmarks[300]
    right_eyebrow_outer = landmarks[336]
    
    # Calculate mouth openness (key for surprise detection)
    mouth_width = abs(mouth_right.x - mouth_left.x)
    mouth_height = abs(upper_lip.y - lower_lip.y)
    mouth_openness = mouth_height / mouth_width if mouth_width > 0 else 0
    
    # Calculate mouth drop (lower lip position)
    mouth_center_y = (mouth_left.y + mouth_right.y) / 2
    lower_lip_drop = abs(lower_lip.y - mouth_center_y) / mouth_width if mouth_width > 0 else 0
    
    # Smile curve calculation (for happy/sad detection)
    lip_center_y = (upper_lip.y + lower_lip.y) / 2
    smile_curve = (mouth_center_y - lip_center_y) / mouth_width if mouth_width > 0 else 0
    
    # Eye aspect ratio and width for surprise detection
    left_eye_height = abs(left_eye_top.y - left_eye_bottom.y)
    right_eye_height = abs(right_eye_top.y - right_eye_bottom.y)
    avg_eye_height = (left_eye_height + right_eye_height) / 2
    
    left_eye_width = abs(left_eye_outer.x - left_eye_inner.x)
    right_eye_width = abs(right_eye_outer.x - right_eye_inner.x)
    avg_eye_width = (left_eye_width + right_eye_width) / 2
    
    # Eye aspect ratio (height/width) - higher for surprise
    eye_aspect_ratio = avg_eye_height / avg_eye_width if avg_eye_width > 0 else 0
    
    # Eyebrow height for surprise (eyebrows raised)
    left_eyebrow_height = abs(left_eyebrow_inner.y - left_eye_top.y)
    right_eyebrow_height = abs(right_eyebrow_inner.y - right_eye_top.y)
    avg_eyebrow_height = (left_eyebrow_height + right_eyebrow_height) / 2
    
    # Improved emotion classification with better surprise detection
    # SURPRISE: Wide eyes + raised eyebrows + open mouth + dropped lower lip
    if (mouth_openness > 0.15 and  # Mouth significantly open
        lower_lip_drop > 0.02 and  # Lower lip dropped
        eye_aspect_ratio > 0.25 and  # Eyes widened
        avg_eyebrow_height > 0.03):  # Eyebrows raised
        confidence = min((mouth_openness + eye_aspect_ratio + avg_eyebrow_height) * 2, 1.0)
        return "surprised", confidence
    
    # HAPPY: Smile curve (negative = upward curve)
    elif smile_curve < -0.003 and mouth_height / mouth_width > 0.05:
        confidence = min(abs(smile_curve) * 200, 1.0)
        return "happy", confidence
    
    # SAD: Frown curve (positive = downward curve)
    elif smile_curve > 0.002:
        confidence = min(smile_curve * 200, 1.0)
        return "sad", confidence
    
    # SLEEPY: Very small eye opening
    elif eye_aspect_ratio < 0.15:
        confidence = min((0.2 - eye_aspect_ratio) * 3, 1.0)
        return "sleepy", confidence
    
    else:
        return "neutral", 0.6

def get_hand_gesture(hand_landmarks):
    if hand_landmarks:
        landmarks = hand_landmarks[0].landmark
        finger_states = []

        finger_states.append(landmarks[4].x < landmarks[3].x)
        for tip_id in [8, 12, 16, 20]:
            finger_states.append(landmarks[tip_id].y < landmarks[tip_id - 2].y)

        if finger_states == [False, False, False, False, False]:
            return "rock"
        elif finger_states == [True, True, True, True, True]:
            return "paper"
        elif finger_states == [False, True, True, False, False]:
            return "scissors"
    return None

def decide_winner(player, computer):
    global player_score, computer_score

    if player == computer:
        return "Draw!"
    elif (player == "rock" and computer == "scissors") or \
         (player == "scissors" and computer == "paper") or \
         (player == "paper" and computer == "rock"):
        player_score += 1
        return "You Win!"
    else:
        computer_score += 1
        return "Computer Wins!"

def get_emotion_reaction(emotion, result):
    """Get a reaction message based on emotion and game result"""
    reactions = {
        "happy": {
            "You Win!": "Great! Your happiness helped you win!",
            "Computer Wins!": "Stay positive! Your smile is still winning!", 
            "Draw!": "Happy with the tie! Keep smiling!"
        },
        "sad": {
            "You Win!": "Cheer up! You won this round!",
            "Computer Wins!": "Don't be sad, try again!",
            "Draw!": "A tie! Maybe that will cheer you up!"
        },
        "surprised": {
            "You Win!": "Surprise! You won!",
            "Computer Wins!": "Surprised by the loss? Try again!",
            "Draw!": "Surprising tie!"
        },
        "sleepy": {
            "You Win!": "Even sleepy, you won!",
            "Computer Wins!": "Wake up for the next round!",
            "Draw!": "Sleepy tie! Need some coffee?"
        },
        "neutral": {
            "You Win!": "Nice win!",
            "Computer Wins!": "Better luck next time!",
            "Draw!": "It's a tie!"
        }
    }
    return reactions.get(emotion, reactions["neutral"]).get(result, result)

print("Starting Rock Paper Scissors game...")
print("Press 'q' to quit, 'r' to restart round, 'l' to toggle landmarks")

# Create window with proper flags
cv2.namedWindow("Rock Paper Scissors", cv2.WINDOW_AUTOSIZE)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from camera")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    # Face detection
    face_results = face_detection.process(rgb)
    face_mesh_results = face_mesh.process(rgb)
    
    # Detect emotion from face mesh
    if face_mesh_results.multi_face_landmarks:
        for face_landmarks in face_mesh_results.multi_face_landmarks:
            current_emotion, emotion_confidence = detect_emotion(face_landmarks)
            
            # Draw face mesh landmarks if toggle is enabled
            if show_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    face_landmarks, 
                    mp_face_mesh.FACEMESH_CONTOURS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1)
                )

    if face_results.detections and round_active:
        for detection in face_results.detections:
            mp_drawing.draw_detection(frame, detection)

        if not countdown_started:
            countdown_start_time = time.time()
            countdown_started = True
            beep()

        elapsed = int(time.time() - countdown_start_time)

        # Countdown overlay with adaptive styling
        if elapsed < countdown_duration:
            countdown_num = countdown_duration - elapsed
            countdown_text = f"GET READY: {countdown_num}"
            
            # Create pulsing effect with adaptive scaling
            pulse_scale = 1.0 + 0.3 * abs(time.time() % 1 - 0.5)
            text_scale_countdown = (2.0 * pulse_scale) * text_scale  # Apply global text scaling
            
            text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_DUPLEX, text_scale_countdown, max(1, int(4 * text_scale)))[0]
            text_x = max(10, (w - text_size[0]) // 2)  # Prevent negative positioning
            text_y = h // 2
            
            # Background circle for countdown (adaptive size)
            circle_radius = max(30, int((text_size[0]//2 + 50) * ui_scale))
            cv2.circle(frame, (text_x + text_size[0]//2, text_y - text_size[1]//2), 
                      circle_radius, (0, 0, 0), -1)
            cv2.circle(frame, (text_x + text_size[0]//2, text_y - text_size[1]//2), 
                      circle_radius, (0, 255, 255), max(1, int(3 * ui_scale)))
            
            # Countdown text with shadow effect (adaptive thickness)
            shadow_thickness = max(1, int(6 * text_scale))
            main_thickness = max(1, int(4 * text_scale))
            
            cv2.putText(frame, countdown_text, (text_x + 2, text_y + 2), 
                       cv2.FONT_HERSHEY_DUPLEX, text_scale_countdown, (0, 0, 0), shadow_thickness)
            cv2.putText(frame, countdown_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_DUPLEX, text_scale_countdown, (0, 255, 255), main_thickness)
            
            if (countdown_duration - elapsed) <= 3:
                beep()
        else:
            # Capture hand gesture
            hand_results = hands.process(rgb)

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # Always draw basic hand landmarks during gesture capture
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    gesture = get_hand_gesture([hand_landmarks])

                    if gesture:
                        player_move = gesture
                        computer_move = random.choice(["rock", "paper", "scissors"])
                        basic_result = decide_winner(player_move, computer_move)
                        result_text = get_emotion_reaction(current_emotion, basic_result)

                        round_active = False
                        countdown_started = False
    
    # Draw hand landmarks outside of game logic if toggle is enabled
    if show_landmarks and not round_active:
        hand_results_always = hands.process(rgb)
        if hand_results_always.multi_hand_landmarks:
            for hand_landmarks in hand_results_always.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
                )

    # Create a more attractive UI overlay with adaptive sizing
    overlay = frame.copy()
    
    # Create semi-transparent background for UI elements (adaptive height)
    cv2.rectangle(overlay, (0, 0), (w, int(overlay_height * ui_scale)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Calculate text sizes based on scaling factor
    score_font_size = 1.0 * text_scale
    emotion_font_size = 0.9 * text_scale
    landmarks_font_size = 0.6 * text_scale
    gesture_font_size = 1.0 * text_scale
    
    # Calculate line spacing based on scale
    line_height = int(35 * ui_scale)
    start_y = int(45 * ui_scale)
    
    # Display scores with adaptive styling
    score_text = f"SCORE - YOU: {player_score}  |  COMPUTER: {computer_score}"
    
    # Check text width and truncate if necessary for low resolution
    if actual_width < 800:
        score_text = f"YOU: {player_score} | PC: {computer_score}"
    
    cv2.putText(frame, score_text, (20, start_y), cv2.FONT_HERSHEY_DUPLEX, 
                score_font_size, (255, 255, 255), max(1, int(3 * text_scale)))
    cv2.putText(frame, score_text, (20, start_y), cv2.FONT_HERSHEY_DUPLEX, 
                score_font_size, (0, 255, 255), max(1, int(2 * text_scale)))
    
    # Display current emotion with adaptive design
    emotion_text = f"EMOTION: {current_emotion.upper()}"
    landmarks_text = f"Landmarks: {'ON' if show_landmarks else 'OFF'} (L)"
    
    # Adjust text for low resolution
    if actual_width < 800:
        emotion_text = f"{current_emotion.upper()}"
        landmarks_text = f"Marks: {'ON' if show_landmarks else 'OFF'}"
    
    y_pos = start_y + line_height
    cv2.putText(frame, emotion_text, (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 
                emotion_font_size, (255, 255, 255), max(1, int(3 * text_scale)))
    cv2.putText(frame, emotion_text, (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 
                emotion_font_size, (255, 100, 255), max(1, int(2 * text_scale)))
    
    y_pos += line_height
    cv2.putText(frame, landmarks_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                landmarks_font_size, (255, 255, 0), max(1, int(2 * text_scale)))

    # Display gestures with adaptive layout
    if player_move:
        y_pos += line_height
        player_text = f"YOU: {player_move.upper()}"
        cv2.putText(frame, player_text, (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 
                    gesture_font_size, (255, 255, 255), max(1, int(3 * text_scale)))
        cv2.putText(frame, player_text, (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 
                    gesture_font_size, (0, 255, 0), max(1, int(2 * text_scale)))
        
    if computer_move:
        y_pos += line_height
        computer_text = f"PC: {computer_move.upper()}" if actual_width < 800 else f"COMPUTER: {computer_move.upper()}"
        cv2.putText(frame, computer_text, (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 
                    gesture_font_size, (255, 255, 255), max(1, int(3 * text_scale)))
        cv2.putText(frame, computer_text, (20, y_pos), cv2.FONT_HERSHEY_DUPLEX, 
                    gesture_font_size, (0, 100, 255), max(1, int(2 * text_scale)))

    # Display result with adaptive styling
    if result_text:
        # Create background for result text (adaptive height)
        result_bg = frame.copy()
        result_area_height = int(120 * ui_scale)
        cv2.rectangle(result_bg, (0, h - result_area_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(result_bg, 0.7, frame, 0.3, 0, frame)
        
        # Adjust result text for different resolutions
        display_result = result_text
        if actual_width < 800 and len(result_text) > 25:
            # Truncate long messages for small screens
            display_result = result_text[:22] + "..."
        
        # Main result text with adaptive sizing
        result_font_size = 1.3 * text_scale
        result_size = cv2.getTextSize(display_result, cv2.FONT_HERSHEY_DUPLEX, result_font_size, max(1, int(3 * text_scale)))[0]
        result_x = max(10, (w - result_size[0]) // 2)  # Prevent negative positioning
        result_y = h - int(70 * ui_scale)
        
        cv2.putText(frame, display_result, (result_x, result_y), cv2.FONT_HERSHEY_DUPLEX, 
                    result_font_size, (255, 255, 255), max(1, int(4 * text_scale)))
        cv2.putText(frame, display_result, (result_x, result_y), cv2.FONT_HERSHEY_DUPLEX, 
                    result_font_size, (0, 255, 255), max(1, int(3 * text_scale)))
        
        # Instructions with adaptive text
        if actual_width < 800:
            instruction_text = "[R] Again | [Q] Quit | [L] Marks"
        else:
            instruction_text = "Press [R] to play again  |  [Q] to quit  |  [L] to toggle landmarks"
        
        inst_font_size = 0.8 * text_scale
        inst_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, inst_font_size, max(1, int(2 * text_scale)))[0]
        inst_x = max(10, (w - inst_size[0]) // 2)  # Prevent negative positioning
        inst_y = h - int(30 * ui_scale)
        
        cv2.putText(frame, instruction_text, (inst_x, inst_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    inst_font_size, (200, 200, 200), max(1, int(2 * text_scale)))
    else:
        # Show game instructions when no result (adaptive)
        if not round_active:
            if actual_width < 800:
                instruction_text = "Show face to start!"
            else:
                instruction_text = "Show your face to start a new round!"
            
            inst_font_size = 1.0 * text_scale
            inst_size = cv2.getTextSize(instruction_text, cv2.FONT_HERSHEY_SIMPLEX, inst_font_size, max(1, int(3 * text_scale)))[0]
            inst_x = max(10, (w - inst_size[0]) // 2)  # Prevent negative positioning
            inst_y = h - int(50 * ui_scale)
            
            cv2.putText(frame, instruction_text, (inst_x, inst_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        inst_font_size, (255, 255, 0), max(1, int(3 * text_scale)))

    # Resize the frame for better display (keeping aspect ratio) - Auto-adjusted based on camera resolution
    aspect_ratio = w / h
    new_height = int(display_width / aspect_ratio)
    frame_resized = cv2.resize(frame, (display_width, new_height))

    # Display the resized frame
    cv2.imshow("Rock Paper Scissors", frame_resized)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        player_move = ""
        computer_move = ""
        result_text = ""
        round_active = True
        countdown_started = False
    elif key == ord('l') or key == ord('L'):
        show_landmarks = not show_landmarks
        print(f"Landmarks display: {'ON' if show_landmarks else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
