import os

files = [
    "D:/iomp/dark_store_project/code/3_gym_environment.py",
    "D:/iomp/dark_store_project/code/6_evaluation.py"
]

for filepath in files:
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()
    
    code = code.replace("max_steps=250", "max_steps=300")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(code)

print("Updated max_steps=300 successfully.")
