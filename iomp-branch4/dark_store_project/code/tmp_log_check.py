import csv
rows = list(csv.DictReader(open('D:/iomp/dark_store_project/results/training_log.csv','r',encoding='utf-8')))
print(f"Total log entries: {len(rows)}")
print(f"\nLearning curve (final run):")
milestones = [0, 29, 59, 99, 149, 199, 249, 299]
for i in milestones:
    if i < len(rows):
        r = rows[i]
        print(f"  Step {r['timestep']:>7}: success={r['success_rate']:>5}%, reward={float(r['avg_reward']):>7.1f}, avg_steps={float(r['avg_length']):>6.1f}")
print(f"\nFinal entry:")
r = rows[-1]
print(f"  Step {r['timestep']}: success={r['success_rate']}%, reward={float(r['avg_reward']):.1f}, avg_steps={float(r['avg_length']):.1f}")
