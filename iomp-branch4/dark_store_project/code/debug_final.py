with open('final_out.txt', 'r', encoding='utf-16') as f:
    lines = f.read()
with open('final_out_ok.txt', 'w', encoding='utf-8') as f:
    f.write(lines)
