from abc import ABC, abstractmethod
import numpy as np

class BarrierFunction(ABC):
    @abstractmethod
    def evaluate(self):
        """Evaluate the barrier function at point x."""
        pass

    @abstractmethod
    def gradient(self):
        """Compute the gradient of the barrier function at point x."""
        pass
    
    @abstractmethod
    def set_state(self, x):
        pass

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi        


class CircularBarrierFunction(BarrierFunction):
    def __init__(self, center, radius: float = 0.5):
        self.center = center
        self.radius = radius
        self.la = 0.5

    def evaluate(self):
        dx = self.x[0] - self.center[0]
        dy = self.x[1] - self.center[1]
        dist2 = dx*dx + dy*dy
        return dist2 - self.radius**2 + self.la * np.cos(self._heading_to_obstacle())

    def gradient(self):
        dx = self.x[0] - self.center[0]
        dy = self.x[1] - self.center[1]
        r2 = dx*dx + dy*dy

        phi = np.arctan2(dy, dx)
        sin_term = np.sin(phi - self.x[2])

        grad_x = 2*dx + self.la * sin_term * (-dy / r2)
        grad_y = 2*dy + self.la * sin_term * (dx / r2)
        grad_theta = self.la * sin_term  

        return np.array([grad_x, grad_y, grad_theta])

    def _heading_to_obstacle(self):
        dx = self.x[0] - self.center[0]
        dy = self.x[1] - self.center[1]
        phi = np.arctan2(dy, dx)
        theta = self.x[2]
        diff = self.normalize_angle(phi - theta)
        return diff

    def set_state(self, x):
        self.x = x


class MapBarrierFunction(BarrierFunction):
    def __init__(self, resolution: float, d_safe: float):
        self.resolution = resolution
        self.d_safe = d_safe

        self.origin = np.array([0.0, 0.0])

        self.dist_map = None
        self.grad_x = None
        self.grad_y = None
        self.x = None

        self.la = 0.2

    def direction(self):
        if self.x[3] < 0:
            self.x[2] += np.pi
        

    def set_origin(self, origin):
        self.origin = origin

    def set_state(self, x):
        self.x = np.asarray(x)

    def set_distance_map(self, dist_map, grad_x, grad_y):
        self.dist_map = dist_map
        self.grad_x = grad_x
        self.grad_y = grad_y

    def _world_to_map(self):
        mx = int((self.x[0] - self.origin[0]) / self.resolution)
        my = int((self.x[1] - self.origin[1]) / self.resolution)
        return mx, my

    def evaluate(self):
        if self.dist_map is None:
            raise ValueError("Distance map not set")

        mx, my = self._world_to_map()

        if 0 <= mx < self.dist_map.shape[1] and 0 <= my < self.dist_map.shape[0]:
            d_pixels = self.dist_map[my, mx]
            d_meters = d_pixels * self.resolution
            return d_meters - self.d_safe + self.la * np.cos(self._heading_to_obstacle())

        # Outside map -> unsafe
        return -1e3

    def _heading_to_obstacle(self):
        self.direction()
        mx, my = self._world_to_map()
        dx = self.grad_x[my, mx]
        dy = self.grad_y[my, mx]
        phi = np.arctan2(dy, dx)
        theta = self.x[2]
        diff = self.normalize_angle(phi - theta)
        return diff


    def gradient(self):
        if self.grad_x is None or self.grad_y is None:
            raise ValueError("Gradient map not set")

        mx, my = self._world_to_map()

        # Spatial gradient of distance transform (meters)
        gx = self.grad_x[my, mx] #/ self.resolution
        gy = self.grad_y[my, mx] #/ self.resolution
        r2 = gx*gx + gy*gy

        # Heading gradient
        diff = self._heading_to_obstacle()
        grad_theta = self.la * np.sin(diff)

        phi = np.arctan2(gy, gx)
        sin_term = np.sin(phi - self.x[2])

        grad_x = 2*gx + self.la * sin_term * (-gy / r2)
        grad_y = 2*gy + self.la * sin_term * (gx / r2)
        grad_theta = self.la * sin_term  

        return np.array([grad_x, grad_y, grad_theta])
