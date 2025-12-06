
import numpy as np

class TrajectoryPlanner:
    """
    基于五次多项式的轨迹规划器
    """
    def __init__(self):
        pass

    def generate_trajectory(self, start_state, end_state, time_duration, dt=0.1):
        """
        生成五次多项式轨迹
        
        Args:
            start_state: 起始状态 [x, y, vx, vy, ax, ay]
            end_state: 目标状态 [x, y, vx, vy, ax, ay]
            time_duration: 规划时长
            dt: 时间步长
            
        Returns:
            list: 轨迹点列表 [(x, y, v, yaw), ...]
        """
        sx, sy, svx, svy, sax, say = start_state
        gx, gy, gvx, gvy, gax, gay = end_state
        
        # 计算x方向系数
        ax = self._quintic_polynomial(sx, svx, sax, gx, gvx, gax, time_duration)
        # 计算y方向系数
        ay = self._quintic_polynomial(sy, svy, say, gy, gvy, gay, time_duration)
        
        trajectory = []
        for t in np.arange(0, time_duration + dt, dt):
            x = self._calc_point(ax, t)
            y = self._calc_point(ay, t)
            vx = self._calc_first_derivative(ax, t)
            vy = self._calc_first_derivative(ay, t)
            
            v = np.sqrt(vx**2 + vy**2)
            yaw = np.arctan2(vy, vx)
            
            trajectory.append((x, y, v, yaw))
            
        return trajectory
    
    def _quintic_polynomial(self, xs, vxs, axs, xe, vxe, axe, T):
        """
        计算五次多项式系数
        x(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        """
        a0 = xs
        a1 = vxs
        a2 = axs / 2.0
        
        A = np.array([
            [T**3, T**4, T**5],
            [3*T**2, 4*T**3, 5*T**4],
            [6*T, 12*T**2, 20*T**3]
        ])
        
        b = np.array([
            xe - a0 - a1*T - a2*T**2,
            vxe - a1 - 2*a2*T,
            axe - 2*a2
        ])
        
        x = np.linalg.solve(A, b)
        
        a3 = x[0]
        a4 = x[1]
        a5 = x[2]
        
        return [a0, a1, a2, a3, a4, a5]
        
    def _calc_point(self, a, t):
        return a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3 + a[4]*t**4 + a[5]*t**5
        
    def _calc_first_derivative(self, a, t):
        return a[1] + 2*a[2]*t + 3*a[3]*t**2 + 4*a[4]*t**3 + 5*a[5]*t**4
