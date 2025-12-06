import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import xml.etree.ElementTree as ET
import traci
import os
import sys
import time
from matplotlib.animation import FuncAnimation

# 设置matplotlib使用独立窗口显示
import matplotlib

matplotlib.use('TkAgg')  # 强制使用TkAgg后端，显示独立窗口

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


class SUMOGridGenerator:
    def __init__(self, net_file_path, sumo_cfg_path=None, verbose=False):#true改成False
        """
        初始化SUMO路网栅格化生成器

        Args:
            net_file_path: SUMO网络文件路径
            sumo_cfg_path: SUMO配置文件路径
            verbose: 是否打印详细日志（False 可提速）
        """
        self.net_file_path = net_file_path
        self.sumo_cfg_path = sumo_cfg_path or "sumo/car.sumocfg"
        self.verbose = verbose  # 控制打印输出
        self.grid_width = 3.5  # 栅格宽度(米) - 对应车道宽度
        self.grid_length = 5.0  # 栅格长度(米) - 沿车道方向

        # E0路段参数(从SUMO文件分析得出，车道宽度3.5米)
        self.e0_start_x = 0.0
        self.e0_end_x = 482.0  # E0路段结束位置(停止线位置)
        self.lane_width = 3.5  # 车道宽度(米)

        # E0路段车道信息 (4个车道，从南到北编号0-3)
        self.num_lanes = 4
        self.e0_lanes = [
            {"index": 0, "y": -12.25},  # 车道0中心线Y坐标
            {"index": 1, "y": -8.75},  # 车道1中心线Y坐标
            {"index": 2, "y": -5.25},  # 车道2中心线Y坐标
            {"index": 3, "y": -1.75}  # 车道3中心线Y坐标
        ]

        # 栅格化范围：从起点到停止线前50米 (0 到 L-50)
        self.grid_start_x = self.e0_start_x  # 0.0米 (起点)
        self.grid_end_x = self.e0_end_x - 47.0  # 435.0米 (停止线前47米)

        # 计算沿车道方向的栅格数量
        self.grid_length_count = int(np.ceil((self.grid_end_x - self.grid_start_x) / self.grid_length))

        # 计算E0路段的y坐标范围(考虑3.5米车道宽度)
        self.min_y = min([lane["y"] for lane in self.e0_lanes]) - self.lane_width / 2  # -14.0
        self.max_y = max([lane["y"] for lane in self.e0_lanes]) + self.lane_width / 2  # 0.0

        self.grids = []
        self.traci_connected = False

        # 颜色映射：车辆数量对应颜色
        self.color_map = {
            0: 'green',  # 无车辆
            1: 'yellow',  # 一辆车
            'many': 'red'  # 多辆车(>1)
        }

    def start_sumo(self, gui=False):
        """
        启动SUMO仿真

        Args:
            gui: 是否使用SUMO GUI (True为有界面，False为后台运行)
        """
        try:
            if self.traci_connected:
                traci.close()
                self.traci_connected = False

            sumo_binary = "sumo-gui" if gui else "sumo"
            sumo_cmd = [sumo_binary, "-c", self.sumo_cfg_path]

            traci.start(sumo_cmd)
            self.traci_connected = True
            # print(f"SUMO started successfully with config: {self.sumo_cfg_path}")
            # print(f"GUI mode: {'ON' if gui else 'OFF'}")

        except Exception as e:
            # print(f"Error starting SUMO: {e}")
            self.traci_connected = False

    def stop_sumo(self):
        """
        停止SUMO仿真
        """
        if self.traci_connected:
            try:
                traci.close()
                self.traci_connected = False
                print("SUMO simulation stopped")
            except Exception as e:
                print(f"Error stopping SUMO: {e}")

    def generate_grids(self):
        """
        生成栅格 - 使用新的坐标系统 (i, j)
        i: 车道索引 (0-3)
        j: 栅格序号 (1到grid_length_count)
        """
        print(f"Generating road network grids with new coordinate system...")
        print(f"Grid parameters: width={self.grid_width}m, length={self.grid_length}m")
        print(f"Grid range: x=[{self.grid_start_x}, {self.grid_end_x}], y=[{self.min_y}, {self.max_y}]")
        print(f"Coordinate system: (lane_index_i, grid_sequence_j)")
        print(f"Lane count: {self.num_lanes}, Grid sequence count: {self.grid_length_count}")

        self.grids = []

        # 遍历每个车道 (i)
        for i in range(self.num_lanes):
            lane_info = self.e0_lanes[i]
            lane_center_y = lane_info["y"]

            # 遍历每个栅格序号 (j)
            for j in range(1, self.grid_length_count + 1):
                # 计算栅格的物理坐标
                x_start = self.grid_start_x + (j - 1) * self.grid_length
                x_end = min(x_start + self.grid_length, self.grid_end_x)
                y_start = lane_center_y - self.lane_width / 2
                y_end = lane_center_y + self.lane_width / 2

                grid = {
                    'coordinate': (i, j),  # 新的坐标系统 (车道索引, 栅格序号)
                    'lane_index': i,
                    'grid_sequence': j,
                    'physical_x_range': (x_start, x_end),
                    'physical_y_range': (y_start, y_end),
                    'physical_center_x': (x_start + x_end) / 2,
                    'physical_center_y': lane_center_y,
                    'width': x_end - x_start,
                    'height': self.lane_width,
                    'lane_info': lane_info,
                    'vehicle_count': 0,  # 栅格内车辆数量
                    'vehicle_ids': []  # 栅格内车辆ID列表
                }

                self.grids.append(grid)

        print(f"Total generated grids: {len(self.grids)}")
        print(f"Grid coordinates range: i=[0, {self.num_lanes - 1}], j=[1, {self.grid_length_count}]")

        return self.grids

    def update_vehicle_positions(self):
        """
        更新栅格中的车辆位置信息
        """
        if not self.traci_connected:
            print("SUMO not connected. Please start SUMO first.")
            return

        # 重置所有栅格的车辆计数
        for grid in self.grids:
            grid['vehicle_count'] = 0
            grid['vehicle_ids'] = []

        # 获取所有车辆ID
        vehicle_ids = traci.vehicle.getIDList()

        # 遍历每辆车
        for veh_id in vehicle_ids:
            try:
                # 获取车辆位置
                position = traci.vehicle.getPosition(veh_id)
                lane_id = traci.vehicle.getLaneID(veh_id)

                # 检查是否在E0路段
                if 'E0' in lane_id:
                    x, y = position

                    # 判断车辆在哪个栅格
                    grid_coord = self._position_to_grid(x, y)
                    if grid_coord:
                        i, j = grid_coord
                        grid = self.get_grid_by_coordinate(i, j)
                        if grid:
                            grid['vehicle_count'] += 1
                            grid['vehicle_ids'].append(veh_id)

            except Exception as e:
                if self.verbose:
                    print(f"Error getting position for vehicle {veh_id}: {e}")

        # 打印车辆分布统计
        total_vehicles = len(vehicle_ids)
        occupied_grids = len([g for g in self.grids if g['vehicle_count'] > 0])
        if self.verbose:
            print(f"Total vehicles: {total_vehicles}, Occupied grids: {occupied_grids}")

    def _position_to_grid(self, x, y):
        """
        根据物理位置(x,y)确定对应的栅格坐标(i,j)
        """
        # 检查是否在栅格化范围内
        if x < self.grid_start_x or x >= self.grid_end_x:
            return None

        # 确定j坐标（栅格序号）
        j = int((x - self.grid_start_x) / self.grid_length) + 1
        if j > self.grid_length_count:
            return None

        # 确定i坐标（车道索引）
        for i, lane_info in enumerate(self.e0_lanes):
            lane_center_y = lane_info["y"]
            lane_y_min = lane_center_y - self.lane_width / 2
            lane_y_max = lane_center_y + self.lane_width / 2

            if lane_y_min <= y <= lane_y_max:
                return (i, j)

        return None

    def get_grid_color(self, vehicle_count):
        """
        根据车辆数量获取栅格颜色
        """
        if vehicle_count == 0:
            return self.color_map[0]  # 绿色
        elif vehicle_count == 1:
            return self.color_map[1]  # 黄色
        else:
            return self.color_map['many']  # 红色

    def visualize_grids_with_vehicles(self, figsize=(12, 8)):
        """
        可视化栅格化路网 - 显示车辆占用情况（静态）
        """
        if not self.grids:
            print("Please generate grids first!")
            return

        # 创建单个图：栅格坐标系统
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # 绘制栅格坐标系统（带车辆信息）
        self._plot_grid_coordinates_with_vehicles(ax)

        plt.tight_layout()
        plt.show()

        # 打印统计信息
        self._print_vehicle_statistics()

    def visualize_real_time(self, figsize=(14, 10), update_interval=100, qminx=None):
        """
        实时可视化栅格化路网 - 动态显示车辆占用情况

        Args:
            figsize: 图表尺寸
            update_interval: 更新间隔(毫秒)
            qminx: QminxNetwork实例，用于车辆状态分析
        """
        if not self.grids:
            print("Please generate grids first!")
            return

        if not self.traci_connected:
            print("SUMO not connected. Please start SUMO first.")
            return

        # 存储QminxNetwork实例
        self.qminx = qminx

        # 创建图形和子图
        self.fig, self.ax = plt.subplots(1, 1, figsize=figsize)

        # 初始化栅格显示
        self.grid_patches = {}
        self.grid_texts = {}
        self._init_real_time_plot()

        # 创建动画
        self.anim = FuncAnimation(
            self.fig,
            self._update_real_time_plot,
            interval=update_interval,
            blit=False,
            repeat=True
        )

        plt.tight_layout()
        plt.show()

    def _init_real_time_plot(self):
        """
        初始化实时绘图
        """
        # 清空坐标轴
        self.ax.clear()

        # 创建栅格矩形和文本
        for grid in self.grids:
            i, j = grid['coordinate']

            # 创建矩形
            rect = Rectangle(
                (j - 0.4, i - 0.4),
                0.8, 0.8,
                linewidth=2,
                edgecolor='black',
                facecolor='green',  # 初始为绿色
                alpha=0.8
            )
            self.ax.add_patch(rect)
            self.grid_patches[(i, j)] = rect

            # 创建文本标签
            text = self.ax.text(
                j, i,
                f"({i},{j})",
                fontsize=6,
                ha='center',
                va='center',
                color='black',
                weight='bold'
            )
            self.grid_texts[(i, j)] = text

        # 设置坐标轴
        self.ax.set_xlim(0.5, self.grid_length_count + 0.5)
        self.ax.set_ylim(-0.5, self.num_lanes - 0.5)
        self.ax.set_xlabel('Grid Sequence j (from road start)', fontsize=12)
        self.ax.set_ylabel('Lane Index i', fontsize=12)
        self.ax.set_title('Real-time Grid Occupancy\n(Green=0 vehicles, Yellow=1 vehicle, Red=>1 vehicles)',
                          fontsize=14, weight='bold')

        # 设置刻度
        self.ax.set_xticks(range(1, self.grid_length_count + 1, max(1, self.grid_length_count // 20)))
        self.ax.set_yticks(range(self.num_lanes))

        # 添加网格
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')

        # 添加车道标签
        for i in range(self.num_lanes):
            self.ax.text(-0.2, i, f'Lane {i}', fontsize=10, ha='right', va='center', weight='bold')

        # 添加图例
        legend_elements = [
            patches.Patch(color='green', alpha=0.8, label='0 vehicles'),
            patches.Patch(color='yellow', alpha=0.8, label='1 vehicle'),
            patches.Patch(color='red', alpha=0.8, label='>1 vehicles')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right')

        # 添加时间显示
        self.time_text = self.ax.text(
            0.02, 0.98, 'Step: 0',
            transform=self.ax.transAxes,
            fontsize=12, weight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )

    def _update_real_time_plot(self, frame):
        """
        更新实时绘图
        """
        try:
            # 执行一步仿真
            traci.simulationStep()
            current_step = traci.simulation.getTime()

            # 更新车辆位置
            self.update_vehicle_positions()

            # 更新QminxNetwork车辆状态并打印l2信息 (每步都执行)
            if hasattr(self, 'qminx'):
                self.qminx.update_vehicle_states()
                # 只在车辆存在时打印信息，避免报错
                if "l2" in self.qminx.get_active_vehicle_ids():
                    self.qminx.print_specific_vehicle_info("l2")

            # 更新栅格颜色和文本
            for grid in self.grids:
                i, j = grid['coordinate']
                vehicle_count = grid['vehicle_count']

                # 更新颜色
                color = self.get_grid_color(vehicle_count)
                self.grid_patches[(i, j)].set_facecolor(color)

                # 更新文本
                if vehicle_count > 0:
                    self.grid_texts[(i, j)].set_text(f"({i},{j})\nV:{vehicle_count}")
                    self.grid_texts[(i, j)].set_fontsize(5)
                else:
                    self.grid_texts[(i, j)].set_text(f"({i},{j})")
                    self.grid_texts[(i, j)].set_fontsize(6)

            # 更新时间显示
            self.time_text.set_text(f'Step: {int(current_step)}')

        except Exception as e:
            print(f"Error in animation update: {e}")
            if hasattr(self, 'anim'):
                self.anim.event_source.stop()

        return list(self.grid_patches.values()) + list(self.grid_texts.values()) + [self.time_text]

    def _plot_grid_coordinates_with_vehicles(self, ax):
        """
        绘制栅格坐标系统 (i,j坐标) - 带车辆占用信息
        """
        # 创建栅格坐标系统的可视化
        for grid in self.grids:
            i, j = grid['coordinate']
            vehicle_count = grid['vehicle_count']

            # 根据车辆数量确定颜色
            color = self.get_grid_color(vehicle_count)

            # 在栅格坐标系统中绘制矩形
            rect = Rectangle(
                (j - 0.4, i - 0.4),  # j-0.4, i-0.4 为左下角
                0.8, 0.8,  # 宽度和高度
                linewidth=2,
                edgecolor='black',
                facecolor=color,
                alpha=0.8
            )
            ax.add_patch(rect)

            # 添加坐标标签和车辆数量
            if vehicle_count > 0:
                ax.text(
                    j, i - 0.1,
                    f"({i},{j})",
                    fontsize=6,
                    ha='center',
                    va='center',
                    color='black',
                    weight='bold'
                )
                ax.text(
                    j, i + 0.1,
                    f"Veh:{vehicle_count}",
                    fontsize=6,
                    ha='center',
                    va='center',
                    color='black',
                    weight='bold'
                )
            else:
                ax.text(
                    j, i,
                    f"({i},{j})",
                    fontsize=6,
                    ha='center',
                    va='center',
                    color='black',
                    weight='bold'
                )

        # 设置坐标轴
        ax.set_xlim(0.5, self.grid_length_count + 0.5)
        ax.set_ylim(-0.5, self.num_lanes - 0.5)
        ax.set_xlabel('Grid Sequence j (from road start)', fontsize=12)
        ax.set_ylabel('Lane Index i', fontsize=12)
        ax.set_title(
            'Grid Coordinate System with Vehicle Occupancy\n(Green=0 vehicles, Yellow=1 vehicle, Red=>1 vehicles)',
            fontsize=14, weight='bold')

        # 设置刻度
        ax.set_xticks(range(1, self.grid_length_count + 1, max(1, self.grid_length_count // 20)))
        ax.set_yticks(range(self.num_lanes))

        # 添加网格
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # 添加车道标签
        for i in range(self.num_lanes):
            ax.text(-0.2, i, f'Lane {i}', fontsize=10, ha='right', va='center', weight='bold')

        # 添加图例
        legend_elements = [
            patches.Patch(color='green', alpha=0.8, label='0 vehicles'),
            patches.Patch(color='yellow', alpha=0.8, label='1 vehicle'),
            patches.Patch(color='red', alpha=0.8, label='>1 vehicles')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    def visualize_grids(self, figsize=(12, 8)):
        """
        可视化栅格化路网 - 使用新的坐标系统 (i,j)
        """
        if not self.grids:
            print("Please generate grids first!")
            return

        # 创建单个图：栅格坐标系统
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # 绘制栅格坐标系统
        self._plot_grid_coordinates(ax)

        plt.tight_layout()
        plt.show()

        # 打印统计信息
        self._print_grid_statistics()

    def _plot_grid_coordinates(self, ax):
        """
        绘制栅格坐标系统 (i,j坐标)
        """
        # 创建栅格坐标系统的可视化
        for grid in self.grids:
            i, j = grid['coordinate']

            # 在栅格坐标系统中绘制矩形
            rect = Rectangle(
                (j - 0.4, i - 0.4),  # j-0.4, i-0.4 为左下角
                0.8, 0.8,  # 宽度和高度
                linewidth=2,
                edgecolor='darkblue',
                facecolor='lightblue',
                alpha=0.7
            )
            ax.add_patch(rect)

            # 添加坐标标签
            ax.text(
                j, i,
                f"({i},{j})",
                fontsize=8,
                ha='center',
                va='center',
                color='darkblue',
                weight='bold'
            )

        # 设置坐标轴
        ax.set_xlim(0.5, self.grid_length_count + 0.5)
        ax.set_ylim(-0.5, self.num_lanes - 0.5)
        ax.set_xlabel('Grid Sequence j (from road start)', fontsize=12)
        ax.set_ylabel('Lane Index i', fontsize=12)
        ax.set_title('Grid Coordinate System\n(i=lane_index, j=grid_sequence)', fontsize=14, weight='bold')

        # 设置刻度
        ax.set_xticks(range(1, self.grid_length_count + 1))
        ax.set_yticks(range(self.num_lanes))

        # 添加网格
        ax.grid(True, alpha=0.5)
        ax.set_aspect('equal')

        # 添加车道标签
        for i in range(self.num_lanes):
            ax.text(-0.2, i, f'Lane {i}', fontsize=10, ha='right', va='center', weight='bold')

    def _print_grid_statistics(self):
        """
        打印栅格统计信息
        """
        print("\n=== Grid Statistics ===")
        print(f"Grid size: {self.grid_width}m × {self.grid_length}m")
        print(
            f"Physical area: X[{self.grid_start_x:.1f}, {self.grid_end_x:.1f}], Y[{self.min_y:.1f}, {self.max_y:.1f}]")
        print(f"Coordinate system: (lane_index_i, grid_sequence_j)")
        print(f"Lane indices: i ∈ [0, {self.num_lanes - 1}]")
        print(f"Grid sequences: j ∈ [1, {self.grid_length_count}]")
        print(f"Total grids: {len(self.grids)} = {self.num_lanes} lanes × {self.grid_length_count} grids/lane")

    def _print_vehicle_statistics(self):
        """
        打印车辆统计信息
        """
        if self.verbose:
            print("\n=== Vehicle Statistics ===")

            # 统计不同车辆数量的栅格
            empty_grids = len([g for g in self.grids if g['vehicle_count'] == 0])
            single_grids = len([g for g in self.grids if g['vehicle_count'] == 1])
            multi_grids = len([g for g in self.grids if g['vehicle_count'] > 1])

            total_vehicles = sum([g['vehicle_count'] for g in self.grids])

            print(f"Empty grids (0 vehicles): {empty_grids} (Green)")
            print(f"Single vehicle grids: {single_grids} (Yellow)")
            print(f"Multiple vehicle grids: {multi_grids} (Red)")
            print(f"Total vehicles in grids: {total_vehicles}")

            # 按车道统计
            print("\nVehicles per lane:")
            for i in range(self.num_lanes):
                lane_vehicles = sum([g['vehicle_count'] for g in self.grids if g['lane_index'] == i])
                print(f"  Lane {i}: {lane_vehicles} vehicles")

    def get_grid_coordinates(self):
        """
        获取所有栅格的坐标信息
        """
        coordinates = []
        for grid in self.grids:
            coord_info = {
                'grid_coordinate': grid['coordinate'],  # (i, j)
                'lane_index': grid['lane_index'],  # i
                'grid_sequence': grid['grid_sequence'],  # j
                'physical_center': (grid['physical_center_x'], grid['physical_center_y']),
                'physical_x_range': grid['physical_x_range'],
                'physical_y_range': grid['physical_y_range'],
                'lane_info': grid['lane_info'],
                'vehicle_count': grid['vehicle_count'],
                'vehicle_ids': grid['vehicle_ids']
            }
            coordinates.append(coord_info)

        return coordinates

    def get_grid_by_coordinate(self, i, j):
        """
        根据栅格坐标(i,j)获取栅格信息
        """
        for grid in self.grids:
            if grid['coordinate'] == (i, j):
                return grid
        return None

    def get_physical_position(self, i, j):
        """
        根据栅格坐标(i,j)获取物理位置
        """
        grid = self.get_grid_by_coordinate(i, j)
        if grid:
            return grid['physical_center_x'], grid['physical_center_y']
        return None

    def simulate_step(self, steps=1):
        """
        执行SUMO仿真步骤并更新车辆位置
        """
        if not self.traci_connected:
            print("SUMO not connected. Please start SUMO first.")
            return False

        try:
            for _ in range(steps):
                traci.simulationStep()

            # 更新车辆位置
            self.update_vehicle_positions()
            return True

        except Exception as e:
            print(f"Error during simulation step: {e}")
            return False


def demo_grid_basic():
    """
    基础栅格功能演示 - 不依赖SUMO
    """
    print("=== Grid Basic Demo ===")

    # 创建栅格生成器
    grid_generator = SUMOGridGenerator("sumo/net.net.xml")

    # 生成栅格
    grids = grid_generator.generate_grids()

    # 显示基础栅格
    grid_generator.visualize_grids()

    # 打印栅格信息
    print(f"Generated {len(grids)} grids")
    print("Grid coordinate examples:")
    for i, grid in enumerate(grids[:10]):
        coord = grid['coordinate']
        center = (grid['physical_center_x'], grid['physical_center_y'])
        print(f"  Grid {coord}: Physical center {center}")


def demo_grid_with_sumo():
    """
    带SUMO集成的栅格演示
    """
    print("=== Grid with SUMO Demo ===")
    print("Note: This demo runs independently. For full system integration, use main.py")

    # SUMO文件路径
    net_file = "sumo/net.net.xml"
    cfg_file = r"D:\desktop\lxpaper\sumo\car.sumocfg"

    # 创建栅格生成器
    grid_generator = SUMOGridGenerator(net_file, cfg_file)

    # 生成栅格
    grids = grid_generator.generate_grids()

    try:
        # 启动SUMO
        grid_generator.start_sumo(gui=True)

        # 运行几个仿真步骤
        print("Running SUMO simulation...")
        for step in range(50):
            success = grid_generator.simulate_step()
            if not success:
                break

            if step % 10 == 0:
                print(f"Simulation step: {step}")

        # 显示最终的栅格状态
        print("\nFinal grid state with vehicles:")
        grid_generator.visualize_grids_with_vehicles()

    except Exception as e:
        print(f"Error during simulation: {e}")

    finally:
        # 停止SUMO
        grid_generator.stop_sumo()


def main():
    """
    Grid模块独立测试
    """
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--with-sumo':
        demo_grid_with_sumo()
    else:
        demo_grid_basic()
        print("\nTo test with SUMO integration, run: python grid.py --with-sumo")
        print("For full system functionality, use: python main.py")


if __name__ == "__main__":
    main()
