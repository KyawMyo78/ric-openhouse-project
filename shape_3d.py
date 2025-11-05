"""
3D Shape Module
Handles 3D shape rendering and transformations
"""

import numpy as np
import cv2
import math


class Shape3D:
    """Base class for 3D shapes"""
    
    def __init__(self, position, size=100):
        """
        Initialize 3D shape
        
        Args:
            position: (x, y, z) position
            size: Size of shape
        """
        self.position = np.array(position, dtype=float)
        self.size = size
        self.rotation = np.array([0.0, 0.0, 0.0])  # Rotation angles (x, y, z)
        self.color = (0, 255, 0)
        self.selected = False
        
    def get_vertices(self):
        """Get vertices of shape - to be overridden"""
        return np.array([])
    
    def get_edges(self):
        """Get edges (vertex pairs) - to be overridden"""
        return []
    
    def rotate(self, dx, dy, dz):
        """Rotate shape"""
        self.rotation[0] += dx
        self.rotation[1] += dy
        self.rotation[2] += dz
    
    def move(self, dx, dy, dz):
        """Move shape"""
        self.position[0] += dx
        self.position[1] += dy
        self.position[2] += dz


class Cube(Shape3D):
    """3D Cube"""
    
    def get_vertices(self):
        """Get cube vertices"""
        s = self.size / 2
        vertices = np.array([
            [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],  # Back face
            [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]       # Front face
        ])
        return vertices
    
    def get_edges(self):
        """Get cube edges (vertex index pairs)"""
        return [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Back face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Front face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Connecting edges
        ]


class Pyramid(Shape3D):
    """3D Pyramid"""
    
    def get_vertices(self):
        """Get pyramid vertices"""
        s = self.size / 2
        h = self.size * 0.8
        vertices = np.array([
            [-s, s, -s], [s, s, -s], [s, s, s], [-s, s, s],  # Base
            [0, -h, 0]  # Apex
        ])
        return vertices
    
    def get_edges(self):
        """Get pyramid edges"""
        return [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Base
            (0, 4), (1, 4), (2, 4), (3, 4)   # Sides to apex
        ]


class Sphere(Shape3D):
    """3D Sphere (rendered as wireframe circles)"""
    
    def __init__(self, position, size=100, detail=16):
        super().__init__(position, size)
        self.detail = detail
    
    def get_vertices(self):
        """Get sphere vertices (latitude and longitude circles)"""
        vertices = []
        r = self.size / 2
        
        # Latitude circles
        for lat in range(-self.detail//2, self.detail//2 + 1):
            theta = (lat / (self.detail//2)) * (math.pi / 2)
            y = r * math.sin(theta)
            radius = r * math.cos(theta)
            
            for lon in range(self.detail):
                phi = (lon / self.detail) * (2 * math.pi)
                x = radius * math.cos(phi)
                z = radius * math.sin(phi)
                vertices.append([x, y, z])
        
        return np.array(vertices)
    
    def get_edges(self):
        """Get sphere edges for wireframe"""
        edges = []
        n = self.detail
        
        # Connect latitude circles
        for i in range(len(self.get_vertices()) - n):
            if (i + 1) % n != 0:
                edges.append((i, i + 1))
            else:
                edges.append((i, i - n + 1))
            edges.append((i, i + n))
        
        return edges


class Renderer3D:
    """Renders 3D shapes to 2D screen"""
    
    def __init__(self, width, height):
        """
        Initialize 3D renderer
        
        Args:
            width: Screen width
            height: Screen height
        """
        self.width = width
        self.height = height
        self.fov = 500  # Field of view
        self.camera_distance = 500
        
    def project_point(self, point):
        """
        Project 3D point to 2D screen
        
        Args:
            point: (x, y, z) 3D coordinates
            
        Returns:
            (x, y) 2D screen coordinates
        """
        x, y, z = point
        z = z + self.camera_distance
        
        if z == 0:
            z = 0.1
        
        # Perspective projection
        factor = self.fov / z
        x_proj = int(x * factor + self.width / 2)
        y_proj = int(y * factor + self.height / 2)
        
        return (x_proj, y_proj)
    
    def transform_vertices(self, vertices, rotation, position):
        """
        Transform vertices (rotate and translate)
        
        Args:
            vertices: Array of 3D vertices
            rotation: (rx, ry, rz) rotation angles in radians
            position: (x, y, z) position
            
        Returns:
            Transformed vertices
        """
        rx, ry, rz = rotation
        
        # Rotation matrices
        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        rot_y = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        rot_z = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # Apply rotations
        vertices = vertices @ rot_x.T @ rot_y.T @ rot_z.T
        
        # Translate
        vertices = vertices + position
        
        return vertices
    
    def render_shape(self, frame, shape):
        """
        Render 3D shape to frame
        
        Args:
            frame: Output frame
            shape: Shape3D object
        """
        vertices = shape.get_vertices()
        edges = shape.get_edges()
        
        # Transform vertices
        transformed = self.transform_vertices(
            vertices,
            shape.rotation,
            shape.position
        )
        
        # Project to 2D
        projected = [self.project_point(v) for v in transformed]
        
        # Draw edges
        color = shape.color if not shape.selected else (255, 0, 255)
        for edge in edges:
            if edge[0] < len(projected) and edge[1] < len(projected):
                pt1 = projected[edge[0]]
                pt2 = projected[edge[1]]
                
                # Check if points are on screen
                if (0 <= pt1[0] < self.width and 0 <= pt1[1] < self.height and
                    0 <= pt2[0] < self.width and 0 <= pt2[1] < self.height):
                    cv2.line(frame, pt1, pt2, color, 2)
        
        # Draw center point if selected
        if shape.selected:
            center_2d = self.project_point(shape.position)
            cv2.circle(frame, center_2d, 8, (255, 255, 0), -1)


class Shape3DManager:
    """Manages 3D shapes"""
    
    def __init__(self, width, height):
        """Initialize 3D shape manager"""
        self.renderer = Renderer3D(width, height)
        self.shapes = []
        self.selected_shape = None
        self.auto_rotate = True
        
    def add_cube(self, position=(0, 0, 0)):
        """Add cube to scene"""
        cube = Cube(position, size=100)
        cube.color = (0, 255, 255)
        self.shapes.append(cube)
        return cube
    
    def add_pyramid(self, position=(0, 0, 0)):
        """Add pyramid to scene"""
        pyramid = Pyramid(position, size=120)
        pyramid.color = (255, 165, 0)
        self.shapes.append(pyramid)
        return pyramid
    
    def add_sphere(self, position=(0, 0, 0)):
        """Add sphere to scene"""
        sphere = Sphere(position, size=80)
        sphere.color = (255, 0, 255)
        self.shapes.append(sphere)
        return sphere
    
    def update(self):
        """Update shapes (auto-rotate if enabled)"""
        if self.auto_rotate:
            for shape in self.shapes:
                shape.rotate(0.01, 0.015, 0.005)
    
    def render_all(self, frame):
        """Render all shapes"""
        for shape in self.shapes:
            self.renderer.render_shape(frame, shape)
    
    def select_shape_at(self, screen_pos):
        """Select shape near screen position"""
        # Deselect all
        for shape in self.shapes:
            shape.selected = False
        
        # Find closest shape
        min_dist = float('inf')
        closest = None
        
        for shape in self.shapes:
            center_2d = self.renderer.project_point(shape.position)
            dist = np.linalg.norm(np.array(screen_pos) - np.array(center_2d))
            if dist < min_dist and dist < 100:
                min_dist = dist
                closest = shape
        
        if closest:
            closest.selected = True
            self.selected_shape = closest
            return True
        
        return False
    
    def clear_all(self):
        """Clear all shapes"""
        self.shapes.clear()
        self.selected_shape = None
