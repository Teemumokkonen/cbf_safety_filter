import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, OccupancyGrid
from python.barrier_function import CircularBarrierFunction, MapBarrierFunction
from tf_transformations import euler_from_quaternion
from cbf import ControlBarrierFunction
import numpy as np
import cv2


class ControlFilter(Node):
    def __init__(self):
        super().__init__('control_filter_node')
        self.barrier_function = CircularBarrierFunction(center=np.array([2.0, -0.5]))
        self.costmap_barrier_function = MapBarrierFunction(
            resolution=0.05,
            d_safe=0.4
        )
        self.cbf = ControlBarrierFunction(alpha=1.0)
        #self.cbf.add_constraint(self.barrier_function)
        self.cbf.add_constraint(self.costmap_barrier_function)

        self.get_logger().info('Control Filter Node has been started.')
        self.twist_subscriber = self.create_subscription(
            Twist,
            "cmd_vel/raw",
            self.twist_callback,
            1
        )

        self.odom_subscriber = self.create_subscription(
            Odometry,
            "odom",
            self.odom_callback,
            1
        )

        self.safe_twist_publisher = self.create_publisher(
            Twist,
            "cmd_vel",
            1
        )

        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/local_costmap/costmap',
            self.costmap_callback,
            10
        )
        self.state_initialized = False

    def twist_callback(self, msg: Twist):
        if not self.state_initialized:
            self.get_logger().warn("State not initialized yet. Ignoring velocity command.", throttle_duration_sec=1.0)
            return
        velocity = [msg.linear.x, msg.angular.z]
        velocity = self.cbf.evaluate(velocity)
        safe_msg = Twist()
        safe_msg.linear.x = velocity[0]
        safe_msg.angular.z = velocity[1]

        self.safe_twist_publisher.publish(safe_msg)

        
    def odom_callback(self, msg: Odometry):
        position = msg.pose.pose.position
        yaw = euler_from_quaternion([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])[2]
        self.cbf.add_rotation(yaw)
        state = np.array([position.x, position.y, yaw, msg.twist.twist.linear.x])
        self.barrier_function.set_state(state)
        self.costmap_barrier_function.set_state(state)
        self.state_initialized = True

    def costmap_callback(self, msg: OccupancyGrid):
        self.get_logger().info("Received costmap update.")
        width = msg.info.width
        height = msg.info.height

        data = np.array(msg.data, dtype=np.int8).reshape((height, width))

        img = np.zeros((height, width), dtype=np.uint8)
        img[data == 100] = 0      # unknown
        img[data < 100] = 255       # free

        img = np.flipud(img)

        obstacle_map = np.zeros_like(img, dtype=np.uint8)
        obstacle_map[img == 0] = 1

        dist = cv2.distanceTransform(255 - obstacle_map * 255, cv2.DIST_L2, 5)

        self.costmap_barrier_function.dist_map = dist
        dy, dx = np.gradient(dist)
        self.costmap_barrier_function.set_distance_map(dist, dx, dy)

        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        self.costmap_barrier_function.set_origin(np.array([origin_x, origin_y]))

def main(args=None):
    rclpy.init(args=args)
    node = ControlFilter()

    rclpy.spin(node)

if __name__ == '__main__':
    main()