import os
import datetime
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

class Potential_avoid():
    """https://qiita.com/koichi_baseball/items/0a6eef85be8d700f6125を参考"""
    def __init__(self, delt, speed, weight_goal):
        self.delt = delt
        self.speed = speed
        self.potential_min = -1
        self.potential_max = 1
        self.weight_obst = 0.1
        self.weight_goal = weight_goal
    
    def cal_potential(self, x, y, goal, obst_target, actions): #actions : [壁weight x, 壁weight y, 壁distance x, 壁distance y, ...]
        tmp_pot = 0
        for obst in obst_target:
            # 障害物がないとき(Noneがはいっている)
            if obst[0] == None or obst[1] == None:
                obst_pot = 0

            # 障害物の座標のpotentialはmax
            elif obst[1] == y and obst[0] == x:
                obst_pot = self.potential_max
            else:
                if obst[2] == 50: # ただの壁(costmapを作成したときに適当に決めた値)
                    obst_weight_x, obst_weight_y, obst_distance_x, obst_distance_y = actions[0], actions[0], 0, 0 # 壁はx,yでわけない
                elif obst[2] == 75: # 車だった
                    obst_weight_x, obst_weight_y, obst_distance_x, obst_distance_y = actions[4], actions[5], actions[6], actions[7] 
                elif obst[2] == 100: # 歩行者だった
                    obst_weight_x, obst_weight_y, obst_distance_x, obst_distance_y = actions[8], actions[9], actions[10], actions[11] 
               
                # print("obst_weight : ", obst_weight)
                obst_pot =  1 / np.sqrt(np.square((x - obst[0] + obst_distance_x)/obst_weight_x) + np.square((y - obst[1] + obst_distance_y)/obst_weight_y))
                # obst_pot += obst_pot * self.weight_obst

            tmp_pot += obst_pot

        # ゴールの座標はpotentialはmin
        if goal[1] == y and goal[0] == x:
            goal_pot = self.potential_min
        else:
            goal_pot = -1 / math.sqrt(pow((y - goal[1]),  2) + pow((x - goal[0]),  2))
        
        pot_all = tmp_pot + actions[12] * goal_pot # actions[6]はゴールのウェイト

        return pot_all
    
    def plot(self, output_route, grid_map, goal, obst_target, obst_grid, actions, now_time, episode):
        # print("actions : ", actions)
        grid_height, grid_width = grid_map.shape[:2]
        pot_map = np.zeros((100, 100))
        
        obst_grid = np.zeros((0))
        for obst in obst_target:
            if obst[2] == 50: # ただの壁(costmapを作成したときに適当に決めた値)
                obst_weight_x, obst_weight_y, obst_distance_x, obst_distance_y = actions[0], actions[0], 0, 0 # 壁はx,yでわけない
            elif obst[2] == 75: # 車だった
                obst_weight_x, obst_weight_y, obst_distance_x, obst_distance_y = actions[4], actions[5], actions[6], actions[7] 
            elif obst[2] == 100: # 歩行者だった
                obst_weight_x, obst_weight_y, obst_distance_x, obst_distance_y = actions[8], actions[9], actions[10], actions[11] 

            obst_pot = 1 / np.sqrt(np.square((grid_map[:, :, 0] - obst[0] + obst_distance_x)/obst_weight_x) + np.square((grid_map[:, :, 1] - obst[1] + obst_distance_y)/obst_weight_y)) # x, yでウェイトを分けたため式の変更
            # print("obst_pot : ", obst_pot.shape)
            # print("obst : ", obst[:2])
            pot_map += obst_pot

            # grid上の物体の位置を記録
            grid_from_pose = np.stack([np.full((grid_height, grid_width), obst[0])
                                      ,np.full((grid_height, grid_width), obst[1])], -1)
            # gridmapとgrid_fram_waypointの差を計算
            error_grid_space = np.sum(np.abs(grid_map[:, :, :2] - grid_from_pose), axis=2)
            # 計算された差から, 一番値が近いグリッドを計算
            nearest_grid_space = np.array(np.unravel_index(np.argmin(error_grid_space), error_grid_space.shape)) # 最小値の座標を取得
            obst_grid = np.block([obst_grid, nearest_grid_space])
        obst_grid = obst_grid.reshape(-1, 2)
        obst_grid[:, 0] = grid_height - obst_grid[:, 0] # 描写のために反転する
        
        # 障害物の位置はmaxにする
        # for o_grid in obst_grid:
        #     pot_map[o_grid[0], o_grid[1]] = 1 * np.sqrt(np.square(obst_weight_x) + np.square(obst_weight_y))
        
        # ゴールの位置から計算
        goal_pot = -1 / np.sqrt(np.square(grid_map[:, :, 0] - goal[0]) + np.square(grid_map[:, :, 1] - goal[1]))

        # 合計を算出
        pot_all = pot_map + actions[6] * goal_pot
        # pot_all  = np.clip(pot_all , -2, 100)

        # 画像に構造を追加
        pot_all_max = np.max(pot_all)
        pot_all_min = np.min(pot_all)
        pot_all_normal = (pot_all - pot_all_min) / (pot_all_max - pot_all_min)
        
        # ルートをプロット
        route_grid = np.zeros((0))
        for route in output_route:
            grid_from_pose = np.stack([np.full((grid_height, grid_width), route[0])
                                      ,np.full((grid_height, grid_width), route[1])], -1)
            # gridmapとgrid_fram_waypointの差を計算
            error_grid_space = np.sum(np.abs(grid_map[:, :, :2] - grid_from_pose), axis=2)
            # 計算された差から, 一番値が近いグリッドを計算
            nearest_grid_space = np.array(np.unravel_index(np.argmin(error_grid_space), error_grid_space.shape)) # 最小値の座標を取得
            route_grid = np.block([route_grid, nearest_grid_space])
        route_grid = route_grid.reshape(-1, 2)
        route_grid[:, 0] = grid_height - route_grid[:, 0] # 描写のために反転する

        plot_t_delta = datetime.timedelta(hours=9)
        plot_JST = datetime.timezone(plot_t_delta, "JST")
        plot_now_JST = datetime.datetime.now(plot_JST)
        plot_now_time = plot_now_JST.strftime("%Y%m%d%H%M%S")

        dict = {"pot_all" : pot_all, "route" : route_grid, "obst_grid" : obst_grid}
        if os.path.exists("./data_{}/potential/ep{}".format(now_time, episode)):
            pass
        else:
            os.makedirs("./data_{}/potential/ep{}".format(now_time, episode))
        np.save("./data_{}/potential/ep{}/potentional_map_{}.npy".format(now_time, episode, plot_now_time), dict)

        # print("pot_all : ", pot_all.shape)

    def calculation(self, start, goal, actions, grid_map, yaw, velocity, change_flag, now_time, episode, first_step):
        actions = actions.numpy()
        # grid_map内の障害物の位置をまとめる
        grid_height, grid_width = grid_map.shape[0], grid_map.shape[1]
        obst_grid = np.array(list(zip(*np.where(grid_map[:, :, 2] > 0)))) # 障害物の場所(grid場所)を計算
        obst_position = np.array([grid_map[position[0], position[1]] for position in obst_grid]) # 障害物のgrid番号と実際のmapから障害物の座標を計算
        xy_pos = start
        output_route = np.array([start[0], start[1], 0, yaw, velocity, change_flag])[np.newaxis, :] # ルートをここに追加していく
        
        count = 0
        goal_flag = False

        while True:
            count += 1
            error2obst = np.sum(np.abs(obst_position[:, :2] - xy_pos), axis=1) # 距離を測る場合、計算速度の都合上マンハッタン距離を使用
            error2obst_sort = np.argsort(error2obst)
            if len(error2obst_sort) > 200:
                nearest_obstacle = np.zeros((200, 3))
            else:
                nearest_obstacle = np.zeros((len(error2obst), 3))
            for i in range(len(nearest_obstacle)): #近い障害物の座標をn個得る
                nearest_obstacle[i] = obst_position[np.where(error2obst_sort == i)[0]]

            vx = -(self.cal_potential(xy_pos[0] + self.delt, xy_pos[1], goal, nearest_obstacle, actions) - self.cal_potential(xy_pos[0], xy_pos[1], goal, nearest_obstacle, actions)) / self.delt
            vy = -(self.cal_potential(xy_pos[0], xy_pos[1] + self.delt, goal, nearest_obstacle, actions) - self.cal_potential(xy_pos[0], xy_pos[1], goal, nearest_obstacle, actions)) / self.delt
            
            v = np.sqrt(vx * vx + vy * vy)

            # 正規化
            vy /= v / self.speed
            vx /= v / self.speed

            xy_pos[0] += vx
            xy_pos[1] += vy

            output_route = np.append(output_route, np.array([xy_pos[0], xy_pos[1], 0, yaw, velocity, change_flag])[np.newaxis, :], axis=0)
            
            # ゴールに近づいた場合，10,000回ループした場合，終了
            if np.abs(goal[0] - xy_pos[0]) < self.speed and np.abs(goal[1] - xy_pos[1]) < self.speed:
                goal_flag = True
                break
            if count > 30:
                print("count over !!!")
                goal_flag = False
                break
        self.plot(output_route, grid_map, goal, nearest_obstacle, obst_grid, actions, now_time, episode)

        return goal_flag, output_route
