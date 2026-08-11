import wandb

# 替换为你的实体名和项目名
api = wandb.Api()
entity = "axi-the-cat"
project = "sta10000 rVLA_Robotwin"

# 获取该实体和项目下的所有实验(runs)
runs = api.runs(f"{entity}/{project}")

# 遍历并打印实验信息
for run in runs:
    print(f"实验名称: {run.name}")          # 实验的名称
    print(f"实验ID: {run.id}")              # 实验的唯一ID，可用于后续查询
    print(f"实验状态: {run.state}")          # 当前状态，如 'finished', 'running', 'failed' 等
    print("-" * 20)
