#!/usr/bin/env python3
from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_msgs.msg import Float64


class ControlLane(Node):

    def __init__(self):
        super().__init__('control_lane')

        self.sub_lane_error = self.create_subscription(
            Float64,
            '/lane_error',
            self.callback_follow_lane,
            1
        )

        self.sub_avoid_cmd = self.create_subscription(
            Twist,
            '/avoid_control',
            self.callback_avoid_cmd,
            1
        )

        self.sub_avoid_active = self.create_subscription(
            Bool,
            '/avoid_active',
            self.callback_avoid_active,
            1
        )

        self.pub_cmd_vel = self.create_publisher(
            Twist,
            '/cmd_vel',
            1
        )

        self.last_error = 0
        self.integral_error = 0

        self.base_speed = 0.6
        self.first_callback = True 

        self.avoid_active = False
        self.avoid_twist = Twist()

    def callback_follow_lane(self, msg):
        if self.avoid_active:
            return

        error = msg.data

        Kp = 1.5 #1.4
        Kd = 0.8 #0.6

        if self.first_callback:
            angular_z = Kp * error
            self.first_callback = False
        else:
            angular_z = Kp * error + Kd * (error - self.last_error)

        self.last_error = error

        speed_factor = max(1 - abs(error), 0) ** 1.5
        
        twist = Twist()

        twist.linear.x = min(max(self.base_speed * speed_factor, -0.85), 0.85)
        twist.angular.z = -max(min(angular_z, 1.0), -1.0)
        self.pub_cmd_vel.publish(twist)

    def callback_avoid_cmd(self, twist_msg):
        self.avoid_twist = twist_msg

        if self.avoid_active:
            self.pub_cmd_vel.publish(self.avoid_twist)

    def callback_avoid_active(self, bool_msg):
        self.avoid_active = bool_msg.data
        if self.avoid_active:
            self.get_logger().info('Avoidance mode activated.')
        else:
            self.get_logger().info('Avoidance mode deactivated. Returning to lane following.')

    def shut_down(self):
        self.get_logger().info('Shutting down. cmd_vel will be 0')
        twist = Twist()
        self.pub_cmd_vel.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = ControlLane()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shut_down()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()