#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String
from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, Twist
import yaml

class CommandParser(Node):
    def __init__(self):
        super().__init__('command_parser')
        
        # Load object-location mappings
        with open('object_locations.yaml', 'r') as f:
            self.object_locations = yaml.safe_load(f)
        self.get_logger().info(f"Loaded {len(self.object_locations)} object mappings")
        
        # Action client for Nav2
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        
        # Publisher for manual movements
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscribe to LLM commands
        self.sub = self.create_subscription(
            String,
            '/llm_commands',
            self.parse_command,
            10
        )
    
    async def parse_command(self, msg):
        """Process LLM output in format 'ACTION|TARGET'"""
        print(msg)
        try:
            action, target = msg.data.lower().split('|')
            self.get_logger().info(f"Executing: {action} -> {target}")
            
            if action == "stop":
                self.stop_robot()
            else:
                await self.handle_movement(target)
                            
        except Exception as e:
            self.get_logger().error(f"Command parsing failed: {str(e)}")
    
    async def handle_movement(self, target):
        """Handle both object navigation and directional movements"""
        # Check for predefined objects
        if target in self.object_locations:
            await self.navigate_to_object(target)
        else:
            # Fallback to manual control
            self.manual_movement(target)
    
    async def navigate_to_object(self, object_name):
        """Send goal to Nav2 server"""
        loc = self.object_locations[object_name]
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'
        goal_msg.pose.pose.position.x = loc['x']
        goal_msg.pose.pose.position.y = loc['y']
        goal_msg.pose.pose.orientation.z = loc['z']
        goal_msg.pose.pose.orientation.w = loc['w']

        
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 action server not available!")
            return
        
        future = await self.nav_client.send_goal_async(goal_msg)
        self.get_logger().info(f"Navigating to {object_name}...")
    
    def manual_movement(self, direction):
        """Handle relative movements via cmd_vel"""
        twist = Twist()
        if "forward" in direction:
            twist.linear.x = 0.2
        elif "back" in direction:
            twist.linear.x = -0.2
        elif "left" in direction:
            twist.angular.z = 0.5
        elif "right" in direction:
            twist.angular.z = -0.5
        else:
            self.get_logger().warn(f"Unknown direction: {direction}")
            return
        
        self.cmd_vel_pub.publish(twist)
        self.get_logger().info(f"Moving {direction}")
    
    def stop_robot(self):
        """Stop all motion"""
        self.cmd_vel_pub.publish(Twist())  # Zero-velocity
        # Cancel any active navigation goals
        if self.nav_client.server_is_ready():
            future = self.nav_client._cancel_all_goals_async()
            rclpy.spin_until_future_complete(self, future)
        self.get_logger().info("Robot stopped")

def main(args=None):
    rclpy.init(args=args)
    parser = CommandParser()
    rclpy.spin(parser)
    parser.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()