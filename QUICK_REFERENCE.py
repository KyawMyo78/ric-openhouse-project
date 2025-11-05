"""
Quick Reference Guide - Virtual Air Drawing

GESTURES CHEAT SHEET:
═══════════════════════════════════════════════════════════════

👆 INDEX FINGER UP
   Action: Draw mode / Place shapes
   Use: Point with index finger and move to draw lines
        or place selected shape

👌 PINCH (Thumb + Index Close)
   Action: Select and move objects
   Use: Pinch fingers together, hover over object to select,
        keep pinching and move to drag object

✋ OPEN PALM (4-5 Fingers Extended)
   Action: Stop current action
   Use: Open hand fully to stop drawing or moving

✌️ TWO-FINGER PINCH (Thumb + Index + Middle)
   Action: Toggle between 2D and 3D mode
   Use: Bring three fingers together for 1 second
        Watch for mode indicator change

👆👆👆 THREE FINGERS UP
   Action: Show shape menu and color palette
   Use: Extend index, middle, and ring fingers
        Menus appear on screen

═══════════════════════════════════════════════════════════════

KEYBOARD SHORTCUTS:
═══════════════════════════════════════════════════════════════

C - Clear canvas (delete all shapes)
H - Toggle help overlay on/off
U - Undo last action
Q - Quit application

═══════════════════════════════════════════════════════════════

2D SHAPES:
═══════════════════════════════════════════════════════════════

Line      - Free-form drawing, follows your finger
Circle    - Place circular shapes
Rectangle - Place rectangular shapes
Triangle  - Place triangular shapes

═══════════════════════════════════════════════════════════════

3D SHAPES:
═══════════════════════════════════════════════════════════════

Cube      - 3D box with auto-rotation
Pyramid   - 3D pyramid with base and apex
Sphere    - 3D sphere (wireframe)

═══════════════════════════════════════════════════════════════

COLORS AVAILABLE:
═══════════════════════════════════════════════════════════════

🟢 Green    🟡 Yellow   🔵 Blue     🔴 Red
🟣 Magenta  🟣 Purple   ⚪ White    🟠 Orange

═══════════════════════════════════════════════════════════════

WORKFLOW EXAMPLES:
═══════════════════════════════════════════════════════════════

📝 Drawing a line:
   1. Show index finger
   2. Move finger in air
   3. Open palm to stop

🎨 Changing color:
   1. Show 3 fingers (menu appears)
   2. Move hand to color palette (bottom)
   3. Hover over desired color
   4. Continue drawing with new color

⭕ Creating shapes:
   1. Show 3 fingers (menu appears)
   2. Move hand to shape menu (right side)
   3. Point at desired shape
   4. Use index finger to place shape

📦 Working with 3D:
   1. Two-finger pinch to toggle 3D mode
   2. Show 3 fingers for shape menu
   3. Select cube/pyramid/sphere
   4. Point with index to place
   5. Watch shapes auto-rotate

🎯 Moving objects:
   1. Pinch gesture (thumb + index)
   2. Hover over object (it highlights)
   3. Keep pinching, move hand
   4. Open palm to release

═══════════════════════════════════════════════════════════════

TIPS FOR BEST RESULTS:
═══════════════════════════════════════════════════════════════

✓ Ensure good lighting
✓ Keep hand centered in frame
✓ Make clear, deliberate gestures
✓ Wait for gesture cooldown (0.5 seconds)
✓ Keep full hand visible (not just fingertips)
✓ Medium distance from camera works best
✓ Avoid rapid gesture switching

═══════════════════════════════════════════════════════════════

PERFORMANCE INDICATORS:
═══════════════════════════════════════════════════════════════

FPS Counter (top-right): Shows frames per second
  • 25-30 FPS: Excellent performance
  • 15-25 FPS: Good performance
  • <15 FPS: Consider closing other apps

Mode Indicator (bottom-right): Shows current mode
  • "Mode: 2D" - Drawing 2D shapes
  • "Mode: 3D" - Working with 3D objects

Gesture Text (top-center): Real-time gesture feedback
  • Shows current recognized gesture
  • Helps confirm gesture detection

═══════════════════════════════════════════════════════════════

TROUBLESHOOTING:
═══════════════════════════════════════════════════════════════

❌ Hand not detected
   → Check lighting
   → Ensure full hand visible
   → Try moving closer/further from camera

❌ Gestures not working
   → Wait for cooldown period
   → Make clearer gestures
   → Check if hand landmarks are visible (green lines)

❌ Drawing is jittery
   → Improve lighting
   → Reduce camera shake
   → Close resource-intensive apps

❌ Low FPS
   → Reduce window size
   → Close other applications
   → Update graphics drivers

═══════════════════════════════════════════════════════════════

ARCHITECTURE OVERVIEW:
═══════════════════════════════════════════════════════════════

┌─────────────────┐
│   draw.py       │ Main application loop
│  (Entry Point)  │ Integrates all modules
└────────┬────────┘
         │
    ┌────┴────────────────────────────────┐
    │                                     │
┌───▼──────────┐                ┌────────▼────────┐
│ hand_tracker │                │  ui_manager     │
│   .py        │                │    .py          │
│              │                │                 │
│ - MediaPipe  │                │ - Buttons       │
│ - Gestures   │                │ - Menus         │
│ - Landmarks  │                │ - Feedback      │
└───┬──────────┘                └─────────────────┘
    │
    │
┌───▼──────────┐                ┌─────────────────┐
│ draw_manager │                │  shape_3d       │
│   .py        │                │    .py          │
│              │                │                 │
│ - 2D Shapes  │                │ - 3D Rendering  │
│ - Objects    │                │ - Projection    │
│ - History    │                │ - Rotation      │
└──────────────┘                └─────────────────┘

═══════════════════════════════════════════════════════════════

Have fun drawing in the air! 🎨✨

═══════════════════════════════════════════════════════════════
"""

if __name__ == "__main__":
    print(__doc__)
