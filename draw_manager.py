"""
Draw Manager Module
Handles drawing operations, object management, and transformations
"""

import cv2
import numpy as np
from collections import deque


class DrawableObject:
    """Base class for drawable objects"""
    
    def __init__(self, shape_type, position, color=(0, 255, 0)):
        """
        Initialize drawable object
        
        Args:
            shape_type: Type of shape ('line', 'circle', 'rectangle', 'triangle')
            position: Starting position (x, y)
            color: BGR color tuple
        """
        self.shape_type = shape_type
        self.position = position  # Center position
        self.color = color
        self.size = 50
        self.initial_size = 50  # Store initial size for scaling
        self.rotation = 0
        self.points = []  # For lines/paths
        self.selected = False
        
    def draw(self, frame):
        """Draw object on frame - to be overridden"""
        pass
    
    def contains_point(self, point):
        """Check if point is inside object - to be overridden"""
        return False
    
    def move(self, delta_x, delta_y):
        """Move object by delta"""
        if self.shape_type == 'line':
            # Move center (like other shapes move position)
            if hasattr(self, 'center') and self.center:
                self.center = (self.center[0] + delta_x, self.center[1] + delta_y)
                self.position = self.center
            # Update working points for selection
            self.points = [(x + delta_x, y + delta_y) for x, y in self.points]
        else:
            self.position = (self.position[0] + delta_x, self.position[1] + delta_y)
    
    def resize(self, scale_factor):
        """Resize object based on scale factor from initial size"""
        self.size = int(self.initial_size * scale_factor)
        # Also update dimensions for rectangles
        if hasattr(self, 'width') and hasattr(self, 'initial_width'):
            self.width = int(self.initial_width * scale_factor)
            self.height = int(self.initial_height * scale_factor)


class Line(DrawableObject):
    """Line/path drawable object"""
    
    def __init__(self, color=(0, 255, 0), thickness=3):
        super().__init__('line', (0, 0), color)
        self.thickness = thickness
        self.original_points = []  # Store original points NEVER modified after drawing ends
        self.original_center = None  # Original center NEVER modified after drawing ends
        self.center = None  # Current center (can move)
        self.size = 1.0  # Use as scale factor for resize
        self.is_drawing = True  # Flag to indicate if still being drawn
        
    def add_point(self, point):
        """Add point to line path"""
        self.points.append(point)
        self.original_points.append(point)
        # Don't update center while drawing - wait until finished
    
    def finish_drawing(self):
        """Called when drawing is complete"""
        self.is_drawing = False
        # Calculate and finalize the center ONLY ONCE
        self._update_center()
        # Only set original_center if not already set
        if self.original_center is None:
            self.original_center = self.center
    
    def _update_center(self):
        """Calculate center point of the line"""
        if len(self.points) > 0:
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            self.center = (sum(xs) // len(xs), sum(ys) // len(ys))
            self.position = self.center
    
    def draw(self, frame):
        """Draw line on frame"""
        # While drawing, use points directly; after drawing, apply transformations
        if self.is_drawing:
            display_points = self.points
        elif len(self.original_points) > 1:
            # Get transformed points (apply resize and rotation)
            display_points = self._get_transformed_points()
        else:
            display_points = self.points
        
        if len(display_points) > 1:
            
            for i in range(1, len(display_points)):
                color = self.color if not self.selected else (255, 0, 255)
                thickness = self.thickness if not self.selected else self.thickness + 2
                cv2.line(frame, display_points[i-1], display_points[i], color, thickness)
            
            # Draw selection handles at start and end
            if self.selected and len(display_points) > 0:
                cv2.circle(frame, display_points[0], 6, (255, 255, 255), -1)  # Start
                cv2.circle(frame, display_points[-1], 6, (255, 255, 255), -1)  # End
    
    def contains_point(self, point):
        """Check if point is near any segment of the line"""
        if len(self.points) < 2:
            return False
        
        px, py = point
        threshold = 15  # Distance threshold for selection
        
        # Check distance to each line segment
        for i in range(1, len(self.points)):
            x1, y1 = self.points[i-1]
            x2, y2 = self.points[i]
            
            # Calculate distance from point to line segment
            # Using formula for point-to-line-segment distance
            line_len_sq = (x2 - x1)**2 + (y2 - y1)**2
            if line_len_sq == 0:
                # Point to point distance
                dist = ((px - x1)**2 + (py - y1)**2) ** 0.5
            else:
                # Project point onto line segment
                t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq))
                proj_x = x1 + t * (x2 - x1)
                proj_y = y1 + t * (y2 - y1)
                dist = ((px - proj_x)**2 + (py - proj_y)**2) ** 0.5
            
            if dist <= threshold:
                return True
        
        return False
    
    def resize(self, scale_factor):
        """Resize line from its center"""
        # Store scale factor, actual transformation happens in _get_transformed_points
        self.size = scale_factor  # Use size to store scale factor
    
    def _get_transformed_points(self):
        """Get points with both resize and rotation applied"""
        if not self.center or len(self.original_points) == 0:
            return self.points
        
        # If original_center is not set yet (still drawing), use current points
        if self.original_center is None:
            return self.points
        
        import math
        
        # Start with original points (NEVER modified)
        transformed = []
        cx, cy = self.center  # Current center (can be moved)
        orig_cx, orig_cy = self.original_center  # Original center (fixed)
        
        # Get scale factor (stored in self.size)
        scale = self.size if hasattr(self, 'size') and self.size != 0 else 1.0
        
        # Get rotation angle
        angle_rad = math.radians(self.rotation) if self.rotation != 0 else 0
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Apply scale and rotation to each original point, then translate to current center
        for px, py in self.original_points:
            # Get relative position from ORIGINAL center
            dx = px - orig_cx
            dy = py - orig_cy
            
            # Apply scale
            dx *= scale
            dy *= scale
            
            # Apply rotation
            if angle_rad != 0:
                new_dx = dx * cos_a - dy * sin_a
                new_dy = dx * sin_a + dy * cos_a
                dx, dy = new_dx, new_dy
            
            # Translate to CURRENT center
            new_x = int(cx + dx)
            new_y = int(cy + dy)
            transformed.append((new_x, new_y))
        
        # Update self.points for contains_point() to work
        self.points = transformed
        return transformed


class Circle(DrawableObject):
    """Circle drawable object"""
    
    def __init__(self, position, color=(0, 255, 255), size=50):
        super().__init__('circle', position, color)
        self.size = size
        self.initial_size = size
        
    def draw(self, frame):
        """Draw circle on frame"""
        color = self.color if not self.selected else (255, 0, 255)
        cv2.circle(frame, self.position, self.size, color, 3)
        if self.selected:
            # Draw selection handles
            cv2.circle(frame, self.position, 5, (255, 255, 255), -1)
    
    def contains_point(self, point):
        """Check if point is inside circle"""
        dx = point[0] - self.position[0]
        dy = point[1] - self.position[1]
        return (dx**2 + dy**2) <= self.size**2


class Rectangle(DrawableObject):
    """Rectangle drawable object"""
    
    def __init__(self, position, color=(255, 0, 255), width=80, height=60):
        super().__init__('rectangle', position, color)
        self.width = width
        self.height = height
        self.initial_width = width
        self.initial_height = height
        
    def draw(self, frame):
        """Draw rectangle on frame"""
        x, y = self.position
        half_w, half_h = self.width // 2, self.height // 2
        
        color = self.color if not self.selected else (255, 0, 255)
        
        # Apply rotation
        if self.rotation != 0:
            corners = np.array([
                [-half_w, -half_h],
                [half_w, -half_h],
                [half_w, half_h],
                [-half_w, half_h]
            ], dtype=np.float32)
            
            angle_rad = np.deg2rad(self.rotation)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            
            rotated = corners @ rotation_matrix.T
            points = (rotated + [x, y]).astype(np.int32)
            cv2.polylines(frame, [points], True, color, 3)
        else:
            pt1 = (x - half_w, y - half_h)
            pt2 = (x + half_w, y + half_h)
            cv2.rectangle(frame, pt1, pt2, color, 3)
        
        if self.selected:
            cv2.circle(frame, self.position, 5, (255, 255, 255), -1)
    
    def contains_point(self, point):
        """Check if point is inside rectangle"""
        x, y = self.position
        half_w, half_h = self.width // 2, self.height // 2
        return (x - half_w <= point[0] <= x + half_w and 
                y - half_h <= point[1] <= y + half_h)
    
    def resize(self, scale_factor):
        """Resize rectangle"""
        if hasattr(self, 'initial_width') and hasattr(self, 'initial_height'):
            self.width = int(self.initial_width * scale_factor)
            self.height = int(self.initial_height * scale_factor)
        else:
            self.width = int(self.width * scale_factor)
            self.height = int(self.height * scale_factor)


class Triangle(DrawableObject):
    """Triangle drawable object"""
    
    def __init__(self, position, color=(0, 165, 255), size=60):
        super().__init__('triangle', position, color)
        self.size = size
        self.initial_size = size
        
    def draw(self, frame):
        """Draw triangle on frame"""
        x, y = self.position
        h = int(self.size * 0.866)  # height of equilateral triangle
        
        color = self.color if not self.selected else (255, 0, 255)
        
        # Define triangle points centered at origin
        corners = np.array([
            [0, -h//2],
            [-self.size//2, h//2],
            [self.size//2, h//2]
        ], dtype=np.float32)
        
        # Apply rotation if needed
        if self.rotation != 0:
            angle_rad = np.deg2rad(self.rotation)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            corners = corners @ rotation_matrix.T
        
        # Translate to position
        points = (corners + [x, y]).astype(np.int32)
        
        cv2.polylines(frame, [points], True, color, 3)
        
        if self.selected:
            cv2.circle(frame, self.position, 5, (255, 255, 255), -1)
    
    def contains_point(self, point):
        """Check if point is inside triangle"""
        dx = abs(point[0] - self.position[0])
        dy = abs(point[1] - self.position[1])
        return dx < self.size//2 and dy < self.size//2


class DrawManager:
    """Manages all drawing operations and objects"""
    
    def __init__(self, width, height):
        """
        Initialize draw manager
        
        Args:
            width: Canvas width
            height: Canvas height
        """
        self.width = width
        self.height = height
        self.objects = []
        self.current_object = None
        self.selected_object = None
        self.mode = 'draw'  # 'draw', 'select', 'move', 'resize'
        self.current_color = (0, 255, 0)
        self.current_shape = 'line'
        self.drawing = False
        
        # For move/resize
        self.last_position = None
        self.initial_size = None
        
        # History for undo
        self.history = deque(maxlen=20)
        
    def start_drawing(self, position):
        """Start drawing new object"""
        if self.current_shape == 'line':
            self.current_object = Line(color=self.current_color)
            self.current_object.add_point(position)
        elif self.current_shape == 'circle':
            self.current_object = Circle(position, color=self.current_color)
            self.objects.append(self.current_object)
            self._save_history()
        elif self.current_shape == 'rectangle':
            self.current_object = Rectangle(position, color=self.current_color)
            self.objects.append(self.current_object)
            self._save_history()
        elif self.current_shape == 'triangle':
            self.current_object = Triangle(position, color=self.current_color)
            self.objects.append(self.current_object)
            self._save_history()
            
        self.drawing = True
    
    def update_drawing(self, position):
        """Update current drawing"""
        if self.drawing and self.current_object:
            if self.current_shape == 'line':
                self.current_object.add_point(position)
    
    def end_drawing(self):
        """Finish drawing"""
        if self.drawing and self.current_object:
            if self.current_shape == 'line' and len(self.current_object.points) > 1:
                # Finalize the line before adding to objects
                if hasattr(self.current_object, 'finish_drawing'):
                    self.current_object.finish_drawing()
                self.objects.append(self.current_object)
                self._save_history()
            self.current_object = None
            self.drawing = False
    
    def select_object(self, position):
        """Select object at position"""
        # Deselect all
        for obj in self.objects:
            obj.selected = False
        
        # Select clicked object (last drawn first)
        for obj in reversed(self.objects):
            if obj.contains_point(position):
                obj.selected = True
                self.selected_object = obj
                print(f"[SELECTED] {obj.shape_type} at position {position}")
                return True
        
        self.selected_object = None
        print(f"[NO SELECTION] No object at position {position}")
        return False
    
    def start_move(self, position):
        """Start moving selected object"""
        if self.selected_object:
            self.last_position = position
            self.mode = 'move'
    
    def update_move(self, position):
        """Update object position during move"""
        if self.selected_object and self.last_position:
            delta_x = position[0] - self.last_position[0]
            delta_y = position[1] - self.last_position[1]
            if abs(delta_x) > 0 or abs(delta_y) > 0:
                self.selected_object.move(delta_x, delta_y)
            self.last_position = position
    
    def end_move(self):
        """Finish moving"""
        self.last_position = None
        self.mode = 'select'
    
    def start_resize(self, position):
        """Start resizing selected object"""
        if self.selected_object:
            self.last_position = position
            self.initial_size = self.selected_object.size if hasattr(self.selected_object, 'size') else 50
            self.mode = 'resize'
    
    def update_resize(self, position):
        """Update object size during resize"""
        if self.selected_object and self.last_position:
            # Calculate scale based on vertical movement
            delta_y = position[1] - self.last_position[1]
            scale = 1.0 + (delta_y / 100.0)
            scale = max(0.5, min(scale, 2.0))  # Clamp scale
            
            self.selected_object.resize(scale)
            self.last_position = position
    
    def end_resize(self):
        """Finish resizing"""
        self.last_position = None
        self.initial_size = None
        self.mode = 'select'
    
    def draw_all(self, frame):
        """Draw all objects on frame"""
        for obj in self.objects:
            obj.draw(frame)
        
        # Draw current object being drawn
        if self.current_object and self.drawing:
            self.current_object.draw(frame)
    
    def clear_canvas(self):
        """Clear all objects"""
        self.objects.clear()
        self.current_object = None
        self.selected_object = None
    
    def undo(self):
        """Undo last action"""
        if self.history:
            self.objects = self.history.pop().copy()
    
    def _save_history(self):
        """Save current state to history"""
        self.history.append(self.objects.copy())
    
    def set_color(self, color):
        """Set current drawing color"""
        self.current_color = color
    
    def set_shape(self, shape):
        """Set current shape type"""
        self.current_shape = shape
