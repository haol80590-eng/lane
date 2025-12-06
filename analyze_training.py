import json
import numpy as np

# Load training data
with open(r'c:\Users\Cindy\Desktop\lxpaper\DRL\ippo\ippo_experiments\ippo_lane_change_20251202_211111\training_stats.json', 'r') as f:
    data = json.load(f)

print('='*80)
print('IPPO Training Results Analysis')
print('='*80)
print(f'\nTotal Episodes: {data["total_episodes"]}')
print(f'Total Steps: {data["total_steps"]}')
print(f'Total Updates: {len(data["total_loss"])}')

# Early training (first 100 episodes)
print('\n' + '='*80)
print('Early Training (Episodes 0-100)')
print('='*80)
early = slice(0, 100)
print(f'Average Reward: {np.mean(data["episode_rewards"][early]):.2f} ± {np.std(data["episode_rewards"][early]):.2f}')
print(f'Success Rate: {np.mean(data["episode_success"][early])*100:.2f}%')
print(f'Dangerous Approaches: {np.mean(data["dangerous_approaches"][early]):.2f} ± {np.std(data["dangerous_approaches"][early]):.2f}')
print(f'Lane Changes: {np.mean(data["lane_changes"][early]):.2f} ± {np.std(data["lane_changes"][early]):.2f}')
print(f'Target Lane Rate: {np.mean(data["target_lane_rates"][early])*100:.2f}%')
print(f'Avg Travel Time: {np.mean(data["avg_travel_times"][early]):.2f} steps')
print(f'Episode Length: {np.mean(data["episode_lengths"][early]):.2f} steps')

# Mid training (episodes 2000-2100)
print('\n' + '='*80)
print('Mid Training (Episodes 2000-2100)')
print('='*80)
mid = slice(2000, 2100)
print(f'Average Reward: {np.mean(data["episode_rewards"][mid]):.2f} ± {np.std(data["episode_rewards"][mid]):.2f}')
print(f'Success Rate: {np.mean(data["episode_success"][mid])*100:.2f}%')
print(f'Dangerous Approaches: {np.mean(data["dangerous_approaches"][mid]):.2f} ± {np.std(data["dangerous_approaches"][mid]):.2f}')
print(f'Lane Changes: {np.mean(data["lane_changes"][mid]):.2f} ± {np.std(data["lane_changes"][mid]):.2f}')
print(f'Target Lane Rate: {np.mean(data["target_lane_rates"][mid])*100:.2f}%')
print(f'Avg Travel Time: {np.mean(data["avg_travel_times"][mid]):.2f} steps')
print(f'Episode Length: {np.mean(data["episode_lengths"][mid]):.2f} steps')

# Late training (last 100 episodes)
print('\n' + '='*80)
print('Late Training (Last 100 Episodes)')
print('='*80)
late = slice(-100, None)
print(f'Average Reward: {np.mean(data["episode_rewards"][late]):.2f} ± {np.std(data["episode_rewards"][late]):.2f}')
print(f'Success Rate: {np.mean(data["episode_success"][late])*100:.2f}%')
print(f'Dangerous Approaches: {np.mean(data["dangerous_approaches"][late]):.2f} ± {np.std(data["dangerous_approaches"][late]):.2f}')
print(f'Lane Changes: {np.mean(data["lane_changes"][late]):.2f} ± {np.std(data["lane_changes"][late]):.2f}')
print(f'Target Lane Rate: {np.mean(data["target_lane_rates"][late])*100:.2f}%')
print(f'Avg Travel Time: {np.mean(data["avg_travel_times"][late]):.2f} steps')
print(f'Episode Length: {np.mean(data["episode_lengths"][late]):.2f} steps')

# Convergence analysis
print('\n' + '='*80)
print('Convergence Analysis')
print('='*80)

# Find when success rate first reaches 90%, 95%, 99%
success_rates = np.array(data["episode_success"])
milestones = [0.5, 0.9, 0.95, 0.99]
for milestone in milestones:
    idx = np.where(success_rates >= milestone)[0]
    if len(idx) > 0:
        print(f'Success rate first reached {milestone*100:.0f}% at episode {idx[0]}')
    else:
        print(f'Success rate never reached {milestone*100:.0f}%')

# Reward convergence
rewards = np.array(data["episode_rewards"])
window = 100
smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
final_reward = np.mean(rewards[-500:])
threshold = final_reward * 0.95  # Within 5% of final performance

converged_idx = np.where(smoothed_rewards >= threshold)[0]
if len(converged_idx) > 0:
    print(f'\nReward converged (within 5% of final) at episode ~{converged_idx[0] + window}')
    print(f'Final reward: {final_reward:.2f}')
    print(f'Convergence threshold: {threshold:.2f}')

# Training stability (last 1000 episodes)
print('\n' + '='*80)
print('Training Stability (Last 1000 Episodes)')
print('='*80)
stable = slice(-1000, None)
print(f'Reward std: {np.std(data["episode_rewards"][stable]):.2f}')
print(f'Success rate std: {np.std(data["episode_success"][stable])*100:.2f}%')
print(f'Dangerous approaches std: {np.std(data["dangerous_approaches"][stable]):.2f}')

# Loss analysis
print('\n' + '='*80)
print('Loss Analysis')
print('='*80)
losses = data["total_loss"]
print(f'Initial loss (first 10 updates): {np.mean(losses[:10]):.4f}')
print(f'Final loss (last 10 updates): {np.mean(losses[-10:]):.4f}')
print(f'Min loss: {np.min(losses):.4f}')
print(f'Max loss: {np.max(losses):.4f}')

# Performance improvement
print('\n' + '='*80)
print('Performance Improvement Summary')
print('='*80)
early_reward = np.mean(data["episode_rewards"][:100])
late_reward = np.mean(data["episode_rewards"][-100:])
early_success = np.mean(data["episode_success"][:100])
late_success = np.mean(data["episode_success"][-100:])
early_dangerous = np.mean(data["dangerous_approaches"][:100])
late_dangerous = np.mean(data["dangerous_approaches"][-100:])
early_lane_changes = np.mean(data["lane_changes"][:100])
late_lane_changes = np.mean(data["lane_changes"][-100:])

print(f'Reward improvement: {early_reward:.2f} -> {late_reward:.2f} ({(late_reward - early_reward):.2f}, {(late_reward/early_reward - 1)*100:.1f}%)')
print(f'Success rate improvement: {early_success*100:.1f}% -> {late_success*100:.1f}% (+{(late_success - early_success)*100:.1f}%)')
print(f'Dangerous approaches reduction: {early_dangerous:.2f} -> {late_dangerous:.2f} ({(late_dangerous - early_dangerous):.2f}, {(late_dangerous/early_dangerous - 1)*100:.1f}%)')
print(f'Lane changes reduction: {early_lane_changes:.1f} -> {late_lane_changes:.1f} ({(late_lane_changes - early_lane_changes):.1f}, {(late_lane_changes/early_lane_changes - 1)*100:.1f}%)')
