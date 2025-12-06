import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.lines as lines
import numpy as np
import os

# 设置全局字体和风格
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_rounded_box(ax, x, y, w, h, color, text=None, subtext=None, text_color='black', alpha=1.0, edge_color='black', linewidth=1.5, fontsize=12):
    """创建一个圆角矩形框"""
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05",
        linewidth=linewidth,
        edgecolor=edge_color,
        facecolor=color,
        alpha=alpha,
        zorder=10
    )
    ax.add_patch(box)
    
    if text:
        ax.text(x + w/2, y + h*0.6, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color=text_color, zorder=20)
    if subtext:
        ax.text(x + w/2, y + h*0.3, subtext, ha='center', va='center', fontsize=fontsize-2, color=text_color, zorder=20)
    
    # 返回连接点坐标
    return {
        'top': (x + w/2, y + h + 0.05),
        'bottom': (x + w/2, y - 0.05),
        'left': (x - 0.05, y + h/2),
        'right': (x + w + 0.05, y + h/2),
        'center': (x + w/2, y + h/2)
    }

def draw_arrow(ax, start, end, color='black', style='->', lw=1.5, curve=0, linestyle='-'):
    """绘制箭头"""
    if curve == 0:
        arrow = patches.FancyArrowPatch(
            start, end,
            arrowstyle=style,
            color=color,
            linewidth=lw,
            mutation_scale=15,
            zorder=5,
            linestyle=linestyle
        )
    else:
        arrow = patches.FancyArrowPatch(
            start, end,
            arrowstyle=style,
            connectionstyle=f"arc3,rad={curve}",
            color=color,
            linewidth=lw,
            mutation_scale=15,
            zorder=5,
            linestyle=linestyle
        )
    ax.add_patch(arrow)

def draw_system_architecture():
    """绘制多智能体系统架构图 (模仿 MAPPO/CPER-MADDPG 风格)"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # 颜色定义 (柔和的学术配色)
    c_env = "#E1F5FE"      # 浅蓝 (环境)
    c_agent = "#FFF3E0"    # 浅橙 (智能体)
    c_actor = "#F8BBD0"    # 粉红 (Actor)
    c_critic = "#C8E6C9"   # 浅绿 (Critic)
    c_buffer = "#E0E0E0"   # 灰色 (经验池)
    c_train = "#D1C4E9"    # 浅紫 (训练模块)
    
    # === 1. SUMO 环境区域 ===
    env_rect = patches.Rectangle((4, 2.5), 4, 5, linewidth=2, edgecolor='#0277BD', facecolor=c_env, alpha=0.5, zorder=1)
    ax.add_patch(env_rect)
    ax.text(6, 7.2, "SUMO Traffic Environment", ha='center', fontsize=16, fontweight='bold', color='#01579B')
    
    # 环境内的元素
    create_rounded_box(ax, 4.5, 6, 3, 0.8, "white", "State Observable Area", "Vehicles, Lanes, Grid", edge_color='#0277BD')
    create_rounded_box(ax, 4.5, 4.5, 3, 0.8, "white", "Physics Engine", "Collision Check, Kinematics", edge_color='#0277BD')
    create_rounded_box(ax, 4.5, 3, 3, 0.8, "white", "Reward Calculation", "Potential Field + TTC", edge_color='#0277BD')
    
    # === 2. 智能体区域 (左侧) ===
    # Agent 1
    ag1 = create_rounded_box(ax, 0.5, 6, 2.5, 1.2, c_agent, "Agent 1", "Vehicle 1", edge_color='#EF6C00')
    # Agent 2
    ag2 = create_rounded_box(ax, 0.5, 4.2, 2.5, 1.2, c_agent, "Agent 2", "Vehicle 2", edge_color='#EF6C00')
    # ...
    ax.text(1.75, 3.6, "...", fontsize=20, fontweight='bold', ha='center')
    # Agent N
    agn = create_rounded_box(ax, 0.5, 2, 2.5, 1.2, c_agent, "Agent N", "Vehicle N", edge_color='#EF6C00')
    
    # === 3. IPPO 训练模块 (底部) ===
    train_rect = patches.Rectangle((2, 0.2), 8, 2, linewidth=2, edgecolor='#512DA8', facecolor=c_train, alpha=0.3, zorder=1)
    ax.add_patch(train_rect)
    ax.text(6, 1.9, "IPPO Centralized Training", ha='center', fontsize=14, fontweight='bold', color='#4527A0')
    
    # 内部组件
    buf = create_rounded_box(ax, 2.5, 0.5, 1.5, 1, c_buffer, "Replay\nBuffer", "Trajectories", edge_color='#616161')
    actor_net = create_rounded_box(ax, 4.5, 0.5, 1.5, 1, c_actor, "Shared\nActor", "Policy $\pi$", edge_color='#AD1457')
    critic_net = create_rounded_box(ax, 6.5, 0.5, 1.5, 1, c_critic, "Shared\nCritic", "Value $V$", edge_color='#2E7D32')
    update_op = create_rounded_box(ax, 8.5, 0.5, 1, 1, "#FFECB3", "PPO\nClip", "Update", edge_color='#FF8F00')
    
    # === 4. 连接线 ===
    
    # 动作输出 (Agent -> Env)
    draw_arrow(ax, ag1['right'], (4, 6.6), color='#E65100', lw=2)
    ax.text(3.5, 6.8, "Action $a_1$", fontsize=10, color='#E65100')
    
    draw_arrow(ax, ag2['right'], (4, 4.8), color='#E65100', lw=2)
    ax.text(3.5, 5.0, "Action $a_2$", fontsize=10, color='#E65100')
    
    draw_arrow(ax, agn['right'], (4, 2.6), color='#E65100', lw=2)
    ax.text(3.5, 2.8, "Action $a_n$", fontsize=10, color='#E65100')
    
    # 状态观测 (Env -> Agent) - 使用曲线
    draw_arrow(ax, (4, 6.2), ag1['right'], color='#01579B', lw=1.5, style='simple', curve=-0.3)
    ax.text(3.2, 6.0, "State $s_1$", fontsize=10, color='#01579B')
    
    draw_arrow(ax, (4, 4.4), ag2['right'], color='#01579B', lw=1.5, style='simple', curve=-0.3)
    
    draw_arrow(ax, (4, 3.0), agn['right'], color='#01579B', lw=1.5, style='simple', curve=-0.3)
    
    # 经验收集 (Agent -> Buffer)
    draw_arrow(ax, (1.75, 2), (2.5, 1.5), color='#424242', lw=2, linestyle='--')
    ax.text(1.8, 1.5, "Experience\n$(s,a,r,s')$", fontsize=10, color='#424242', ha='center')
    
    # 训练流
    draw_arrow(ax, buf['right'], actor_net['left'], color='#4527A0', lw=2)
    draw_arrow(ax, buf['right'], critic_net['left'], color='#4527A0', lw=2, curve=0.5)
    
    # 梯度更新
    draw_arrow(ax, update_op['left'], critic_net['right'], color='red', lw=2)
    draw_arrow(ax, critic_net['left'], actor_net['right'], color='red', lw=2)
    
    # 参数同步 (Training -> Agents)
    # 画一条大的虚线箭头从 Actor 指向左侧的 Agent 区域
    ax.annotate('', xy=(0, 5), xytext=(4.5, 1.5),
                arrowprops=dict(arrowstyle='->', linestyle='dashed', color='#AD1457', lw=2, connectionstyle="angle,angleA=90,angleB=180,rad=10"))
    ax.text(0.2, 8, "Parameter Sync", fontsize=12, fontweight='bold', color='#AD1457', rotation=90)

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figure_system_arch_pro.png'), dpi=300, bbox_inches='tight')
    print("Generated: figure_system_arch_pro.png")
    plt.close()

def draw_detailed_network():
    """绘制详细神经网络结构图 (模仿 SAC/Transformer 结构图)"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # 颜色
    c_input = "#E3F2FD"
    c_grid = "#FFF8E1"
    c_fc = "#F3E5F5"
    c_out = "#E8F5E9"
    
    # === 1. 输入层 ===
    # 状态向量分解
    create_rounded_box(ax, 1, 10, 2, 1, c_input, "Ego State", "3-dim\n(Lane, Target)", edge_color='#1565C0')
    create_rounded_box(ax, 4, 10, 2, 1, c_input, "Grid Map", "15-dim\n(3x5 Binary)", edge_color='#1565C0')
    create_rounded_box(ax, 7, 10, 2, 1, c_input, "Interaction", "5-dim\n(Intent, TTC)", edge_color='#1565C0')
    
    ax.text(5, 11.5, "Input Observation (23-dim)", ha='center', fontsize=16, fontweight='bold')
    
    # === 2. 特征提取层 (Feature Extraction) ===
    feat_box = patches.FancyBboxPatch((0.5, 7), 9, 2, boxstyle="round,pad=0.2", fc="#FAFAFA", ec="#9E9E9E", lw=2, linestyle='--')
    ax.add_patch(feat_box)
    ax.text(1.5, 8.8, "Feature Extractor", fontsize=12, color="#616161", fontweight='bold')
    
    fc1 = create_rounded_box(ax, 3, 7.5, 4, 1, c_fc, "Fully Connected Layer 1", "128 units + ReLU", edge_color='#7B1FA2')
    
    # === 3. 共享表示层 (Shared Representation) ===
    fc2 = create_rounded_box(ax, 3.5, 5.5, 3, 1, c_fc, "Shared Layer 2", "64 units + ReLU", edge_color='#7B1FA2')
    
    # === 4. 双头输出 (Actor & Critic) ===
    # Actor Head (左侧)
    actor_bg = patches.FancyBboxPatch((0.5, 0.5), 4, 4, boxstyle="round,pad=0.2", fc="#FFEBEE", ec="#C62828", lw=2)
    ax.add_patch(actor_bg)
    ax.text(2.5, 4.2, "Actor Head (Policy)", ha='center', fontsize=14, fontweight='bold', color="#C62828")
    
    act_fc = create_rounded_box(ax, 1, 2.5, 3, 0.8, "white", "FC Layer", "32 units", edge_color='#C62828')
    act_out = create_rounded_box(ax, 1.5, 1, 2, 0.8, "#FFCDD2", "Softmax", "Action Probabilities", edge_color='#C62828')
    
    # Critic Head (右侧)
    critic_bg = patches.FancyBboxPatch((5.5, 0.5), 4, 4, boxstyle="round,pad=0.2", fc="#E8F5E9", ec="#2E7D32", lw=2)
    ax.add_patch(critic_bg)
    ax.text(7.5, 4.2, "Critic Head (Value)", ha='center', fontsize=14, fontweight='bold', color="#2E7D32")
    
    crt_fc = create_rounded_box(ax, 6, 2.5, 3, 0.8, "white", "FC Layer", "32 units", edge_color='#2E7D32')
    crt_out = create_rounded_box(ax, 6.5, 1, 2, 0.8, "#C8E6C9", "Linear", "State Value V(s)", edge_color='#2E7D32')
    
    # === 5. 连接线 ===
    # Input -> FC1
    draw_arrow(ax, (2, 10), (3.5, 8.5), lw=1.5)
    draw_arrow(ax, (5, 10), (5, 8.5), lw=1.5)
    draw_arrow(ax, (8, 10), (6.5, 8.5), lw=1.5)
    
    # FC1 -> FC2
    draw_arrow(ax, (5, 7.5), (5, 6.5), lw=2)
    
    # FC2 -> Heads
    draw_arrow(ax, (5, 5.5), (2.5, 3.3), lw=2, color='#C62828') # To Actor
    draw_arrow(ax, (5, 5.5), (7.5, 3.3), lw=2, color='#2E7D32') # To Critic
    
    # Inside Heads
    draw_arrow(ax, (2.5, 2.5), (2.5, 1.8), lw=1.5, color='#C62828')
    draw_arrow(ax, (7.5, 2.5), (7.5, 1.8), lw=1.5, color='#2E7D32')
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'figure_network_detail_pro.png'), dpi=300, bbox_inches='tight')
    print("Generated: figure_network_detail_pro.png")
    plt.close()

if __name__ == "__main__":
    print("Generating professional paper figures...")
    try:
        draw_system_architecture()
        draw_detailed_network()
        print("All professional figures generated successfully!")
    except Exception as e:
        print(f"Error generating figures: {e}")
