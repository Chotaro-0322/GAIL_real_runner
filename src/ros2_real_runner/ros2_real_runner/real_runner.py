from distutils.log import error
from optparse import check_builtin
import os
import sys
from tabnanny import check
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from geometry_msgs.msg import PoseStamped, Pose
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Imu, PointCloud2, Image
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32, Int32, Float32MultiArray, MultiArrayDimension
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge

from .brain import Actor, Critic, Discriminator, Brain
from .potential_f import Potential_avoid
from .ros2_numpy.ros2_numpy.point_cloud2 import pointcloud2_to_array
from .ros2_numpy.ros2_numpy.occupancy_grid import occupancygrid_to_numpy
from .PyTorch_YOLOv3.pytorchyolo import detect
from .PyTorch_YOLOv3.pytorchyolo import models

import numpy as np
import quaternion
import torch
import json
import time
import threading
import ctypes
import pandas as pd
import csv
import datetime
import threading
from tqdm import tqdm
import glob
import random
import cv2

MAX_STEPS = 500
NUM_EPISODES = 1000
NUM_PROCESSES = 1
NUM_ADVANCED_STEP = 50
NUM_COMPLETE_EP = 8

os.chdir("/home/seniorcar/Desktop/yamamoto/Obstacle_avoidance/ros2_RL/src/ros2_real_runner/ros2_real_runner")
print("current pose : ", os.getcwd())

t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(t_delta, "JST")
now_JST = datetime.datetime.now(JST)
now_time = now_JST.strftime("%Y%m%d%H%M%S")
# os.makedirs("./data_{}/weight".format(now_time))
# os.makedirs("./data_{}/potential".format(now_time))
buffer_file_dir = "./data_{}/buffer".format(now_time)
# os.makedirs(buffer_file_dir)

# Python program raising
# exceptions in a python
# thread

def calculate_dot(rot_matrix, coordinate_array):
    result_array = np.zeros((coordinate_array, 3))
    for i, array in enumerate(coordinate_array):
        result_array[i] = np.dot(rot_matrix, array)
    return result_array

  
class Runner(Node):
    def __init__(self):
        super().__init__("real_runner")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ROS setting
        self.current_pose = PoseStamped()
        self.imu = Imu()
        self.closest_waypoint = Int32()

        # self.current_pose_sub = self.create_subscription(PoseStamped, "current_pose", self.current_poseCallback, 1)
        self.imu_sub = self.create_subscription(Imu, "imu_raw",  self.imuCallback, 1)
        self.closest_waypoint_sub = self.create_subscription(Int32, "closest_waypoint", self.closestWaypointCallback, 1)

        self.initialpose_pub = self.create_publisher(PoseWithCovarianceStamped, "initialpose", 1)
        self.lookahead_pub = self.create_publisher(Float32, "minimum_lookahead_distance", 1)

        # その他Flagや設定
        self.initialpose_flag = False # ここの値がTrueなら, initialposeによって自己位置が完了したことを示す。
        self.gridmap_sub_flag = False # ここの値がTrueなら, grid_mapのトピックをサブスクライブできたことを示す。
        self.on_collision_flag = False # このまま行くと障害物にぶつかるというのを格納するための変数
        self.complete_episode_num = 0
        self.penalty_num = 0
        self.error2object  = 1000 # 障害物までの距離をここに入れる(初期値は大きく取りたいので1000mにしている)
        self.check_waypoint_length = 15 # 何m先の障害物を避ける対象としてみなすか(waypointの数)
        self.avoid_margin = 6 # 障害物の何waypoint後をゴールとして設定するか？(waypointの数)
        self.goal_margin = 4 # 経路の何m以内に近づいたら終了とするか
        self.closest_skip_num = 2 # 何個先のwaypoints以上を出力するか?

        self.waypoints = pd.read_csv("/home/seniorcar/Desktop/yamamoto/route/default_waypoints.csv", header=None, skiprows=1).to_numpy()
        
        self.base_expert_waypoints = self.waypoints # ここに全体の経路を控えておく
        self.goal_position = self.base_expert_waypoints[-1, :2].copy()
        
        self.global_start = self.base_expert_waypoints[0, :2].copy()
        self.global_goal = self.base_expert_waypoints[-1, :2].copy()

        self.map_offset = [43, 28.8, 6.6] # マップのズレを試行錯誤で治す！ ROS[x, y, z] ↔ Unity[z, -x, y]
        self.rotation_offset = [0, 0, 10]
        self.quaternion_offset = quaternion.from_rotation_vector(np.array(self.rotation_offset))
        # print("waypoint : \n", self.waypoint)
        # print("GPU is : ", torch.cuda.is_available())

        self.n_in = 1 # 状態
        self.n_out = 3 #行動
        action_num = 3
        self.n_mid = 32

        self.num_states = 1 # ニューラルネットワークの入力数
        self.num_actions = 2 # purepursuit 0.5mと2m

        self.obs_shape = [100, 100, 1]
        # 壁・車・人の順番 x [potential_weight_x, potential_weight_y, x_offset, y_offset]
        # self.actor_up = torch.tensor([0.2, 0.2, 0.0, 0.0, 5.0, 5.0, 0.5, 0.5, 5.0, 5.0, 0.5, 0.5, 100]).to(self.device)
        # self.actor_down = torch.tensor([0.0000, 0.0000, -0.0, -0.0, 0.2, 0.2, -0.5, -0.5, 0.2, 0.2, -0.5, -0.5, 80]).to(self.device)
        self.actor_up = torch.tensor([0.2, 0.2, 0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 3.0, 3.0, 0.5, 0.5, 100]).to(self.device)
        self.actor_down = torch.tensor([0.0000, 0.0000, -0.0, -0.0, 0.2, 0.2, -0.5, -0.5, 0.2, 0.2, -0.5, -0.5, 80]).to(self.device)
        self.actor_limit_high = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100])
        self.actor_limit_low = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 80])
        self.actor_value = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 80])
        self.actor = Actor(self.n_in, self.n_mid, self.n_out, self.actor_up, self.actor_down).to(self.device)
        self.critic = Critic(self.obs_shape[2], self.obs_shape[0],self.obs_shape[1]).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.reward_buffer = 0

        self.global_brain = Brain(self.actor, self.critic, self.discriminator, buffer_file_dir)

        self.current_obs = torch.zeros(NUM_PROCESSES, self.obs_shape[2], self.obs_shape[0], self.obs_shape[1]) # torch size ([16, 4])
        self.episode_rewards = torch.zeros([NUM_PROCESSES, 1]) # 現在の施行の報酬を保持
        self.final_rewards = torch.zeros([NUM_PROCESSES, 1]) # 最後の施行の報酬を保持
        self.old_action_probs = torch.zeros([NUM_ADVANCED_STEP, NUM_PROCESSES, 1]) # ratioの計算のため
        self.obs_np = np.zeros([NUM_PROCESSES, self.obs_shape[2], self.obs_shape[0], self.obs_shape[1]]) 
        self.reward_np = np.zeros([NUM_PROCESSES, 1])
        self.done_np = np.zeros([NUM_PROCESSES, 1])
        self.each_step = np.zeros(0) # 各環境のstep数を記録
        self.episode = 0 # 環境0の施行数

        # ポテンシャル法の部分
        self.Potential_avoid = Potential_avoid(delt=0.2, speed=0.5, weight_goal=30) # weight_obstはディープラーニングで計算

        self.pcd_as_numpy_array = np.zeros((0))
        self.gridmap_object_value = np.zeros((0))
        self.current_pose = np.zeros((2))
        self.num_closest_waypoint = 0
        self.gridmap = np.zeros((0))
        self.grid_resolution = 0
        self.grid_width = 0
        self.grid_height = 0
        self.grid_position = Pose()
        self.vehicle_grid = np.zeros((2))
        self.goal_grid = np.zeros((2))

        self.length_judge_obstacle = 10 # この数字分, 先のwaypointを使って障害物を判断

        self.pcd_subscriber = self.create_subscription(PointCloud2, "clipped_cloud", self.pointcloudCallback, 1)
        self.costmap_subscriber = self.create_subscription(OccupancyGrid, "lgsvl_occupancy_grid", self.costmapCallback, 1)
        self.ndt_pose_subscriber = self.create_subscription(PoseStamped, "ndt_pose", self.ndtPoseCallback, 1)
        self.img_subscriber = self.create_subscription(Image, "image_raw", self.imgCallback, 1)
        
        self.waypoint_publisher = self.create_publisher(Float32MultiArray, "route_waypoints_multiarray", 1)
        self.base_waypoint_publisher = self.create_publisher(Float32MultiArray, "base_route_waypoints_multiarray", 1)
        self.lgsvl_obj_publisher = self.create_publisher(Float32MultiArray, "lgsvl_obj", 1)
        self.yolov3_obj_publisher = self.create_publisher(Float32MultiArray, "yolov3_obj", 1)
        self.closest_waypoint_publisher = self.create_publisher(Int32, "closest_waypoint", 1)
        self.npc_marker_publisher = self.create_publisher(MarkerArray, "npc_position", 1)
        self.image_pub = self.create_publisher(Image, "person_box_image", 1)

        # Yolov3の設定
        self.model = models.load_model(
            "/home/seniorcar/Desktop/yamamoto/Obstacle_avoidance/ros2_RL/src/ros2_real_runner/ros2_real_runner/PyTorch_YOLOv3/config/yolov3.cfg",
            "/home/seniorcar/Desktop/yamamoto/Obstacle_avoidance/ros2_RL/src/ros2_real_runner/ros2_real_runner/PyTorch_YOLOv3/weights/yolov3.weights"
        )
        self.box_msg = Float32MultiArray()
        self._bridge = CvBridge()

        # GAIL(Actor)のweight_path
        weight_path = "/home/seniorcar/Desktop/yamamoto/weight/scenario5_gan/episode_60_finish.pth"
        check_points = torch.load(weight_path)
        print("check_points : ", check_points)
        self.global_brain.main_actor = check_points

    def imuCallback(self, msg):
        self.imu = msg

    def closestWaypointCallback(self, msg):
        self.closest_waypoint = msg.data
    
    def pointcloudCallback(self, msg):
        self.pcd_as_numpy_array = pointcloud2_to_array(msg)
    
    def imgCallback(self, msg):
        cv_img = self._bridge.imgmsg_to_cv2(msg, "rgb8")
        boxes = detect.detect_image(self.model, cv_img).tolist()

        self.publish_coord(boxes)
        self.publish_center_image(cv_img, boxes)

    def ndtPoseCallback(self, msg):
        self.current_pose = np.array([msg.pose.position.x, msg.pose.position.y])
        self.initialpose_flag = True

    def publish_coord(self, boxes):
        """
        座標データをROS送信用に1次元に変更し、ストライドを設定した後publish
        boxes : list [[x1, y1, x2, y2, confidence, class]]
        self.box_msg : Float32MultiArray() 高さと幅のストライドを持つ1次元配列
        """
        # print("boxes : ", boxes)
        bbox_dim1 = []
        for box in boxes:
            if (box[5] == 0) or (box[5] == 1) or (box[5] == 2):
                centroid = [((box[0] + box[2]) / 2), ((box[1] + box[3]) / 2), box[5]]

                for bx in centroid:
                        bbox_dim1.append(bx)
                
        if bbox_dim1:
            self.box_msg.data = bbox_dim1
            dim0 = MultiArrayDimension()
            dim0.label = "foo"
            dim0.size = int(len(self.box_msg.data)/3)
            dim0.stride = 3
            dim1 = MultiArrayDimension()
            dim1.label = "bar"
            dim1.size = 3
            dim1.stride = 1

            self.box_msg.layout.dim = [dim0, dim1]

            self.yolov3_obj_publisher.publish(self.box_msg)

    def publish_center_image(self, img, boxes):
        height, width, _ = img.shape
        for p, box in enumerate(boxes):
            if (box[5] == 0) or (box[5] == 1) or (box[5] == 2):
                img = cv2.circle(img, (np.int((box[0] + box[2])/2), np.int((box[1] + box[3])/2)), 10, (255, 0, 0), -1)
        
        self.image_pub.publish(self._bridge.cv2_to_imgmsg(img, "rgb8"))

    def costmapCallback(self, msg):
        self.grid_resolution = msg.info.resolution
        self.grid_width = msg.info.width
        self.grid_height = msg.info.height
        self.grid_origin = msg.info.origin
        self.gridmap_object_value = occupancygrid_to_numpy(msg)[:, :, np.newaxis] # あとで座標情報とくっつけるために次元を増やしておく
        orientation_array =  np.array([self.grid_origin.orientation.w, self.grid_origin.orientation.x, self.grid_origin.orientation.y, self.grid_origin.orientation.z])
        orientation_quaternion = quaternion.as_quat_array(orientation_array)
        grid_rot_matrix = quaternion.as_rotation_matrix(orientation_quaternion) # 回転行列の作成
        # 回転行列に平行移動を追加する
        grid_rot_matrix[0, 2] = self.grid_origin.position.x
        grid_rot_matrix[1, 2] = self.grid_origin.position.y

        self.grid_coordinate_array = np.zeros((self.grid_height, self.grid_width, 3))
        self.grid_coordinate_array[:, :, 2] = 1
        for i in range(self.grid_width): # /mapの座標を設定 : x軸 
            self.grid_coordinate_array[:, i, 0] = i * self.grid_resolution
        for i in range(self.grid_height): # /mapの座標を設定 : y軸
            self.grid_coordinate_array[i, :, 1] = i * self.grid_resolution

        # 先ほど作成した変形行列(回転+平行移動)を適用する
        self.grid_coordinate_array = self.grid_coordinate_array.reshape(-1, 3)
        
        tmp_transformed_vector = np.zeros((self.grid_coordinate_array.shape[0], 3))
        start = time.time()
        for i, array in enumerate(self.grid_coordinate_array):
            tmp_transformed_vector[i] = np.dot(grid_rot_matrix, array)
        self.grid_coordinate_array = tmp_transformed_vector.reshape(self.grid_height, self.grid_width, 3)

        self.gridmap = np.block([self.grid_coordinate_array[:, :, :2], self.gridmap_object_value]) # [x, y, class]

        self.gridmap_sub_flag = True

    def publish_waypoints(self):
        publish_waypoints = self.waypoints[self.closest_skip_num:]
        multiarray = Float32MultiArray()
        multiarray.layout.dim.append(MultiArrayDimension())
        multiarray.layout.dim.append(MultiArrayDimension())
        multiarray.layout.dim[0].label = "height"
        multiarray.layout.dim[1].label = "width"
        multiarray.layout.dim[0].size = publish_waypoints.shape[0]
        multiarray.layout.dim[1].size = publish_waypoints.shape[1]
        multiarray.layout.dim[0].stride = publish_waypoints.shape[0] * publish_waypoints.shape[1]
        multiarray.layout.dim[1].stride = publish_waypoints.shape[1]
        multiarray.data = publish_waypoints.reshape(1, -1)[0].tolist()
        self.waypoint_publisher.publish(multiarray)
    
    def check_collision_flag(self): # 障害物の回避をするべきかどうかを判断
        # closest_waypointを探索
        error2waypoint = np.sum(np.abs(self.waypoints[:, :2] - self.current_pose), axis=1) # 距離を測る場合、計算速度の都合上マンハッタン距離を使用
        print("error2waypoint : ", error2waypoint)
        self.defaultroute_closest_waypoint = error2waypoint.argmin()
        
        obj_count = 0

        obj_number_list = []
        """ 1. ここのfor文で障害物の有無を判断 """
        for check_num in range(self.check_waypoint_length): 
            if len(self.waypoints) > self.defaultroute_closest_waypoint+check_num:
                grid_from_currentpose = np.stack([np.full((self.grid_height, self.grid_width), self.waypoints[self.defaultroute_closest_waypoint+check_num][0])
                                        ,np.full((self.grid_height, self.grid_width), self.waypoints[self.defaultroute_closest_waypoint+check_num][1])], -1)
                # gridmapとgrid_fram_waypointの差を計算
                error_grid_space = np.sum(np.abs(self.gridmap[:, :, :2] - grid_from_currentpose), axis=2)
                # 計算された差から, 一番値が近いグリッドを計算
                nearest_grid_space = np.unravel_index(np.argmin(error_grid_space), error_grid_space.shape) # 最小値の座標を取得
                self.grid_class = self.gridmap[nearest_grid_space[0], nearest_grid_space[1]][2]
                if self.grid_class != 0: # 何かしらの物体のとき
                    obj_count += 1
                    obj_number_list.append(check_num)
            else:
                pass

        """ 2. 物体が存在しないときは, 目標点の近くにいるかどうかをチェック """
        if obj_count == 0:
            error2goal = np.linalg.norm(self.goal_position - self.current_pose)
            if error2goal < self.goal_margin: # ゴール付近まで来ていたら　waypointsをもとのやつに戻す & goal_positionを修正
                error2waypoint = np.sum(np.abs(self.base_expert_waypoints[:, :2] - self.current_pose), axis=1) # 距離を測る場合、計算速度の都合上マンハッタン距離を使用
                self.defaultroute_closest_waypoint = error2waypoint.argmin()
                self.waypoints = self.base_expert_waypoints[self.defaultroute_closest_waypoint:]
                self.goal_position = self.base_expert_waypoints[-1, :2] # 全経路中の最終点をgoal_positionとする
            else: # ゴール付近に来ていない(避けてる最中 or 普通にもとの経路を走行しているのみ)
                error2waypoint = np.sum(np.abs(self.waypoints[:, :2] - self.current_pose), axis=1) # 距離を測る場合、計算速度の都合上マンハッタン距離を使用
                self.defaultroute_closest_waypoint = error2waypoint.argmin()
                self.waypoints = self.waypoints[self.defaultroute_closest_waypoint:]
            return False # ポテンシャル法を使用せず, 今の経路をひたすら走行し続ける

        else: # 物体が存在するとき
            last_obj_waypoint = obj_number_list[-1]
            obj_pos = self.waypoints[self.defaultroute_closest_waypoint+last_obj_waypoint][:2] # 最後の物体のxyz座標を計算
            error2waypoint = np.sum(np.abs(self.base_expert_waypoints[:, :2] - obj_pos), axis=1) # 全体の経路中, 物体がどの場所にいるかを計算
            obj_waypoint = error2waypoint.argmin()
            if len(self.base_expert_waypoints) > obj_waypoint + self.avoid_margin:
                self.goal_position = self.base_expert_waypoints[obj_waypoint+self.avoid_margin, :2] # 全体の経路の物体の位置のself.aboid_margin後のwaypointsをゴール地点として設定
            else:
                self.goal_position = self.base_expert_waypoints[-1, :2]
            return True # ポテンシャル法を使用する必要がある

    def env_feedback(self, actions, episode, first_step = False):
        # time.sleep(0.1)
        "--------------closest_waypointsおよび、gridmap内の自車位置の予測------------"
        # closest_waypointを探索
        error2waypoint = np.sum(np.abs(self.waypoints[:, :2] - self.current_pose), axis=1) # 距離を測る場合、計算速度の都合上マンハッタン距離を使用
        closest_waypoint = error2waypoint.argmin()
        closest_waypoint_msg = Int32()
        closest_waypoint_msg.data = int(closest_waypoint)
        self.closest_waypoint_publisher.publish(closest_waypoint_msg)

        grid_from_currentpose = np.stack([np.full((self.grid_height, self.grid_width), self.current_pose[0])
                                      ,np.full((self.grid_height, self.grid_width), self.current_pose[1])], -1)
        # gridmapとgrid_fram_waypointの差を計算
        error_grid_space = np.sum(np.abs(self.gridmap[:, :, :2] - grid_from_currentpose), axis=2)
        # 計算された差から, 一番値が近いグリッドを計算
        nearest_grid_space = np.unravel_index(np.argmin(error_grid_space), error_grid_space.shape) # 最小値の座標を取得
        self.vehicle_grid = self.gridmap[nearest_grid_space[0], nearest_grid_space[1]][:2]

        "-------------ポテンシャル法による回避-----------"
        vehicle_position = np.array(self.vehicle_grid)
        # 経路が終盤付近で"out of bounds for axis"のエラーが出るのを防ぐ
        yaw = self.waypoints[closest_waypoint, 3]
        velocity = 3.7
        change_flag = self.waypoints[0, 5]
        goal_flag, output_route = self.Potential_avoid.calculation(vehicle_position, self.goal_position, actions, self.gridmap, yaw, velocity, change_flag, now_time, episode, first_step)
        self.waypoints = output_route
        # """------------経路をpublish------------"""
        # 出力された経路と最も近いウェイポイントを計算
        error2waypoint = np.sum(np.abs(self.waypoints[:, :2] - self.current_pose), axis=1) # 距離を測る場合、計算速度の都合上マンハッタン距離を使用
        closest_waypoint = error2waypoint.argmin()
        self.waypoints = self.waypoints[closest_waypoint:]

    def pandas_init(self):
        self.record = pd.DataFrame({"current_pose_x" : [self.current_pose[0]], "current_pose_y" : [self.current_pose[1]], "current_pose_z" : [0], "error2object" : [0],
                                    "reward" : [0], "reward_mean" : [0], 
                                    "action_wall_x" : [0], "action_wall_y" : [0],
                                    "action_vehicle_x ": [0], "action_vehicle_y ": [0],
                                    "action_human_x" :  [0], "action_human_y" :  [0],
                                    "reward_dist_vehicle2goal" : [0], "reward_discriminator_output" : [0],
                                    "reward_error2expwaypoint" : [0], "reward_on_collision_flag" : [0],
                                    "reward_achive_goal": [0],  "goal_flag" : [0]})

    def run(self):
        while True:
            if (self.initialpose_flag and self.gridmap_sub_flag) is False:
                continue
            collision_flag = self.check_collision_flag()
            print("collision flag : ", collision_flag)
            if collision_flag:
                with torch.no_grad():
                    state_stack = torch.zeros(4, self.grid_height, self.grid_width)
                    for i in range(4):
                        state_stack[i] = torch.from_numpy(self.gridmap_object_value.transpose(2, 0, 1))
                    state_stack = torch.unsqueeze(state_stack, 0).type(torch.FloatTensor).to(self.device)
                    action, _ = self.actor.act(state_stack, 0)
                actions = action[0].cpu().type(torch.FloatTensor)

                self.env_feedback(actions=actions, episode=0, first_step=False) # errorを次のobsercation_next(次の状態)として扱う

            else:
                pass

            self.publish_waypoints()
