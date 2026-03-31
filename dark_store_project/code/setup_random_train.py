with open("D:/iomp/dark_store_project/code/4_ppo_training.py", "r", encoding='utf-8') as f:
    code = f.read()

code = code.replace("optimized_layout.json", "random_layout.json")
code = code.replace("ppo_model.zip", "ppo_model_random.zip")
code = code.replace("training_log.csv", "training_log_random.csv")
code = code.replace("TOTAL_TIMESTEPS = 500_000", "TOTAL_TIMESTEPS = 300_000")
code = code.replace("max_steps=250", "max_steps=300")

with open("D:/iomp/dark_store_project/code/train_random_baseline.py", "w", encoding='utf-8') as f:
    f.write(code)
    
print("Successfully generated train_random_baseline.py")
