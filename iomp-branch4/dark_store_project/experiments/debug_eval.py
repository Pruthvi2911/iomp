import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = BASE_DIR / 'code'
sys.path.append(str(CODE_DIR))
import importlib
from stable_baselines3 import PPO

print('debug start')
gym_env_module = importlib.import_module('3_gym_environment')
DarkStoreEnv = gym_env_module.DarkStoreEnv
print('import done')
model = PPO.load(str(BASE_DIR / 'experiments' / 'exp_lr_0.001' / 'ppo_model.zip'), device='cpu')
print('model loaded')
env = DarkStoreEnv(str(BASE_DIR / 'results' / 'optimized_layout.json'), max_steps=250, max_items_per_order=3)
print('env created')
obs, info = env.reset()
print('reset done', type(obs), info)
action, _ = model.predict(obs, deterministic=True)
print('action', action)
