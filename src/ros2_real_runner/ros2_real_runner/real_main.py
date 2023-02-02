import rclpy
from rclpy.node import Node
import threading

from .real_runner import Runner

def main(args=None):
    rclpy.init(args=args)
    Env = Runner()

    rl_running_thread = threading.Thread(target=Env.run)
    rl_running_thread.start()
    
    rclpy.spin(Env)
    

if __name__ == "__main__":
    main()
