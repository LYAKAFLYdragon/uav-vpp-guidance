import csv
from collections import defaultdict

def analyze_pilot(csv_path):
    results = defaultdict(lambda: defaultdict(list))
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            task = row['task']
            opponent = row['opponent_stage']
            results[task][opponent].append({
                'win': row['win'].lower() == 'true',
                'loss': row['loss'].lower() == 'true',
                'success': row['success'].lower() == 'true',
                'survived': row['survived'].lower() == 'true',
                'termination': row['termination_reason'],
                'steps': int(row['steps']),
                'reward': float(row['total_reward']),
                'final_range': float(row['final_range_m']),
                'final_ata': float(row['final_abs_ata_deg']),
            })
    
    for task in sorted(results.keys()):
        for opponent in sorted(results[task].keys()):
            episodes = results[task][opponent]
            n = len(episodes)
            wins = sum(1 for e in episodes if e['win'])
            losses = sum(1 for e in episodes if e['loss'])
            successes = sum(1 for e in episodes if e['success'])
            survived = sum(1 for e in episodes if e['survived'])
            crashes = sum(1 for e in episodes if 'crash' in e['termination'])
            timeouts = sum(1 for e in episodes if 'timeout' in e['termination'])
            avg_steps = sum(e['steps'] for e in episodes) / n
            avg_reward = sum(e['reward'] for e in episodes) / n
            avg_final_range = sum(e['final_range'] for e in episodes) / n
            avg_final_ata = sum(e['final_ata'] for e in episodes) / n
            
            print(f"{task} vs {opponent}: n={n}, win_rate={wins/n:.2f}, loss_rate={losses/n:.2f}, success_rate={successes/n:.2f}, survival={survived/n:.2f}, crash={crashes/n:.2f}, timeout={timeouts/n:.2f}")
            print(f"  avg_steps={avg_steps:.1f}, avg_reward={avg_reward:.1f}, avg_final_range={avg_final_range:.0f}m, avg_final_ata={avg_final_ata:.1f}deg")

print("=== EXPERT ===")
analyze_pilot("outputs/jsbsim_hrl_comparison/multihead_expert_10seed/summary.csv")
print("\n=== END_TO_END ===")
analyze_pilot("outputs/jsbsim_hrl_comparison/multihead_e2e_10seed/summary.csv")
