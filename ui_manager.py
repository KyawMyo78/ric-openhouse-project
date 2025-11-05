"""
UI Manager Module
Handles on-screen UI elements, menus, and visual feedback
"""

import cv2
import numpy as np


class Button:
    """UI Button class"""
    
    def __init__(self, x, y, width, height, text, color=(100, 100, 100)):
        """
        Initialize button
        
        Args:
            x, y: Button position (top-left)
            width, height: Button dimensions
            text: Button text
            color: Button color (BGR)
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.color = color
        self.hover_color = (150, 150, 150)
        self.active_color = (0, 255, 0)
        self.active = False
        self.visible = True
        self.is_hovered = False
        
    def draw(self, frame):
        """Draw button on frame"""
        if not self.visible:
            return
        
        color = self.active_color if self.active else (self.hover_color if self.is_hovered else self.color)
        
        # Draw button rectangle
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), 
                     color, -1)
        cv2.rectangle(frame, (self.x, self.y), 
                     (self.x + self.width, self.y + self.height), 
                     (255, 255, 255), 2)
        
        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(self.text, font, 0.5, 1)[0]
        text_x = self.x + (self.width - text_size[0]) // 2
        text_y = self.y + (self.height + text_size[1]) // 2
        
        cv2.putText(frame, self.text, (text_x, text_y), 
                   font, 0.5, (255, 255, 255), 1)
    
    def contains(self, point):
        """Check if point is inside button"""
        x, y = point
        return (self.x <= x <= self.x + self.width and 
                self.y <= y <= self.y + self.height)


class ColorPalette:
    """Color selection palette"""
    
    def __init__(self, x, y):
        """Initialize color palette"""
        self.x = x
        self.y = y
        self.colors = [
            (0, 255, 0),      # Green
            (0, 255, 255),    # Yellow
            (255, 0, 0),      # Blue
            (0, 0, 255),      # Red
            (255, 0, 255),    # Magenta
            (128, 0, 128),    # Purple
            (255, 255, 255),  # White
            (0, 165, 255)     # Orange
        ]
        self.button_size = 40
        self.selected_index = 0
        self.visible = True  # Visible by default
        
    def draw(self, frame):
        """Draw color palette"""
        if not self.visible:
            return
        
        for i, color in enumerate(self.colors):
            x = self.x + i * (self.button_size + 5)
            y = self.y
            
            # Draw color box
            cv2.rectangle(frame, (x, y), 
                         (x + self.button_size, y + self.button_size), 
                         color, -1)
            
            # Draw border (thicker if selected)
            thickness = 3 if i == self.selected_index else 1
            cv2.rectangle(frame, (x, y), 
                         (x + self.button_size, y + self.button_size), 
                         (255, 255, 255), thickness)
    
    def select_color_at(self, point):
        """Select color at point"""
        if not self.visible:
            return None
        
        x, y = point
        for i in range(len(self.colors)):
            btn_x = self.x + i * (self.button_size + 5)
            btn_y = self.y
            
            if (btn_x <= x <= btn_x + self.button_size and 
                btn_y <= y <= btn_y + self.button_size):
                self.selected_index = i
                return self.colors[i]
        
        return None
    
    def get_selected_color(self):
        """Get currently selected color"""
        return self.colors[self.selected_index]


class ShapeMenu:
    """Shape selection menu"""
    
    def __init__(self, x, y):
        """Initialize shape menu"""
        self.x = x
        self.y = y
        self.shapes = ['line', 'circle', 'rectangle', 'triangle']
        self.button_size = 60
        self.selected_shape = 'line'
        self.visible = True  # Visible by default
        
        # Create buttons dictionary for easier click detection
        self.buttons = {}
        self._create_buttons()
    
    def _create_buttons(self):
        """Create button objects for each shape"""
        for i, shape in enumerate(self.shapes):
            x = self.x
            y = self.y + i * (self.button_size + 10)
            self.buttons[shape] = Button(x, y, self.button_size, self.button_size, shape)
            if shape == self.selected_shape:
                self.buttons[shape].active = True
        
    def draw(self, frame):
        """Draw shape menu"""
        if not self.visible:
            return
        
        shapes_to_draw = self.shapes
        
        for i, shape in enumerate(shapes_to_draw):
            x = self.x
            y = self.y + i * (self.button_size + 10)
            
            # Draw background
            color = (0, 255, 0) if shape == self.selected_shape else (80, 80, 80)
            cv2.rectangle(frame, (x, y), 
                         (x + self.button_size, y + self.button_size), 
                         color, -1)
            cv2.rectangle(frame, (x, y), 
                         (x + self.button_size, y + self.button_size), 
                         (255, 255, 255), 2)
            
            # Draw shape icon
            center_x = x + self.button_size // 2
            center_y = y + self.button_size // 2
            
            if shape == 'line':
                cv2.line(frame, (x + 10, y + 10), (x + 50, y + 50), (255, 255, 255), 2)
            elif shape == 'circle':
                cv2.circle(frame, (center_x, center_y), 20, (255, 255, 255), 2)
            elif shape == 'rectangle':
                cv2.rectangle(frame, (x + 15, y + 15), (x + 45, y + 45), (255, 255, 255), 2)
            elif shape == 'triangle':
                pts = np.array([[center_x, y + 10], [x + 10, y + 50], [x + 50, y + 50]])
                cv2.polylines(frame, [pts], True, (255, 255, 255), 2)
    
    def select_shape_at(self, point):
        """Select shape at point"""
        if not self.visible:
            return None
        
        x, y = point
        
        for i, shape in enumerate(self.shapes):
            btn_y = self.y + i * (self.button_size + 10)
            
            if (self.x <= x <= self.x + self.button_size and 
                btn_y <= y <= btn_y + self.button_size):
                self.selected_shape = shape
                return shape
        
        return None


class UIManager:
    """Manages all UI elements"""
    
    def __init__(self, width, height):
        """
        Initialize UI manager
        
        Args:
            width: Screen width
            height: Screen height
        """
        self.width = width
        self.height = height
        
        # Create UI elements
        self.clear_button = Button(10, 10, 80, 40, "Clear", (150, 50, 50))
        self.tool_button = Button(100, 10, 100, 40, "Draw", (50, 150, 50))
        
        # Color palette at TOP center
        self.color_palette = ColorPalette(width // 2 - 180, 10)
        self.shape_menu = ShapeMenu(width - 80, 80)
        
        # State
        self.current_tool = 'draw'  # 'draw', 'select', 'move'
        self.gesture_text = ""
        self.show_help = True
        
    def draw(self, frame):
        """Draw all UI elements"""
        # Draw buttons
        self.clear_button.draw(frame)
        self.tool_button.draw(frame)
        
        # Draw color palette
        self.color_palette.draw(frame)
        
        # Draw shape menu
        self.shape_menu.draw(frame)
        
        # Draw gesture feedback
        if self.gesture_text:
            cv2.putText(frame, self.gesture_text, (self.width // 2 - 100, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Draw help text
        if self.show_help:
            help_texts = [
                "RIGHT HAND ONLY:",
                "1 Finger = Draw",
                "Pinch = Select/Move",
                "3 Fingers = Menu, Palm = Stop",
                "",
                "TWO HANDS:",
                "Both Pinch = Resize",
                "Right Pinch + Left L = Rotate",
                "",
                "Keys: C=Clear H=Help U=Undo"
            ]
            
            y_offset = 80
            for text in help_texts:
                cv2.putText(frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                y_offset += 22
        
    
    def handle_button_click(self, point):
        """Handle button clicks"""
        if self.clear_button.contains(point):
            return 'clear'
        elif self.tool_button.contains(point):
            self.toggle_tool()
            return 'tool_toggle'
        
        return None
    
    def toggle_tool(self):
        """Toggle between tools"""
        tools = ['draw', 'select', 'move']
        current_index = tools.index(self.current_tool)
        next_index = (current_index + 1) % len(tools)
        self.current_tool = tools[next_index]
        self.tool_button.text = self.current_tool.capitalize()
    
    def set_gesture_text(self, text):
        """Set gesture feedback text"""
        self.gesture_text = text
    
    def toggle_help(self):
        """Toggle help visibility"""
        self.show_help = not self.show_help
