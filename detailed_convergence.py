import json
import numpy as np
import matplotlib.pyplot as plt

# Load training data
with open(r'c:\Users\Cindy\Desktop\lxpaper\DRL\ippo\ippo_experiments\ippo_lane_change_20251202_211111\training_stats.json', 'r') as f:
    data = json.load(f)

# Check convergence with smoothing
success_rates = np.array(data["episode_success"])
window = 50  # Use smoothing window

# Apply moving average
smoothed_success = np.convolve(success_rates, np.ones(window)/window, mode='valid')

print('='*80)
print('Detailed Convergence Analysis (with smoothing)')
print('='*80)

milestones = [0.5, 0.8, 0.9, 0.95, 0.99]
for milestone in milestones:
    idx = np.where(smoothed_success >= milestone)[0]
    if len(idx) > 0:
        episode = idx[0] + window
        print(f'Success rate (smoothed) reached {milestone*100:.0f}% at episode ~{episode}')
        # Show actual success rate at that point
        actual_rate = np.mean(success_rates[max(0, episode-50):episode+50])
        print(f'  Actual success rate around episode {episode}: {actual_rate*100:.1f}%')

# Check for sudden jumps
print('\n' + '='*80)
print('Sudden Performance Jumps Detection')
print('='*80)

# Check reward jumps
rewards = np.array(data["episode_rewards"])
reward_diff = np.diff(rewards)
large_jumps = np.where(np.abs(reward_diff) > 100)[0]
if len(large_jumps) > 0:
    print(f'\nLarge reward jumps (>100) found at episodes:')
    for jump_idx in large_jumps[:10]:  # Show first 10
        print(f'  Episode {jump_idx}: {rewards[jump_idx]:.1f} -> {rewards[jump_idx+1]:.1f} (delta: {reward_diff[jump_idx]:.1f})')

# Check success rate pattern around episode 358
print('\n' + '='*80)
print('Success Rate Pattern Around Episode 358')
print('='*80)
for i in range(340, 380):
    if i < len(success_rates):
        print(f'Episode {i}: success={success_rates[i]:.1f}, reward={rewards[i]:.1f}, lane_changes={data["lane_changes"][i]:.1f}')

# Analyze episode lengths
print('\n' + '='*80)
print('Episode Length Analysis')
print('='*80)
episode_lengths = np.array(data["episode_lengths"])
print(f'Early episodes (0-100): {np.mean(episode_lengths[:100]):.1f} steps')
print(f'Mid episodes (2000-2100): {np.mean(episode_lengths[2000:2100]):.1f} steps')
print(f'Late episodes (last 100): {np.mean(episode_lengths[-100:]):.1f} steps')
print(f'\nMax episode length: {np.max(episode_lengths):.0f} steps')
print(f'Min episode length: {np.min(episode_lengths):.0f} steps')

# Check if early termination is happening
early_termination = episode_lengths < 100
print(f'\nEpisodes with early termination (<100 steps): {np.sum(early_termination)} ({np.sum(early_termination)/len(episode_lengths)*100:.1f}%)')

# Travel time vs episode length
print('\n' + '='*80)
print('Travel Time vs Episode Length')
print('='*80)
travel_times = np.array(data["avg_travel_times"])
print(f'Early: travel_time={np.mean(travel_times[:100]):.1f}, episode_length={np.mean(episode_lengths[:100]):.1f}')
print(f'Late: travel_time={np.mean(travel_times[-100:]):.1f}, episode_length={np.mean(episode_lengths[-100:]):.1f}')
print(f'Difference: episode_length - travel_time = {np.mean(episode_lengths[-100:]) - np.mean(travel_times[-100:]):.1f} steps')
