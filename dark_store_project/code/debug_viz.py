import sys

with open('viz_out.txt', 'r', encoding='utf-16') as f:
    lines = f.readlines()

with open('debug_viz.txt', 'w', encoding='utf-8') as f:
    f.writelines(lines[-20:])
