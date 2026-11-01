# %% [markdown]
# # Train ACT on a tidy-flywheel dataset (Google Colab, GPU)
#
# Paste this file into Colab cells (each `# %%` is a cell), or run it as a script.
# Set `HF_TOKEN` in Colab secrets (key `HF_TOKEN`), never write it into this file.
#
# Inputs : a LeRobotDataset on the Hub (private is fine), e.g. `takerunakao/tidy-v0`
# Outputs: a checkpoint pushed to the Hub, e.g. `takerunakao/tidy-act-v0`, plus the train log.

# %%
import os, subprocess, sys

DATASET = os.environ.get("DATASET", "takerunakao/tidy-v0")
MODEL = os.environ.get("MODEL", "takerunakao/tidy-act-v0")
STEPS = int(os.environ.get("STEPS", "20000"))
BATCH = int(os.environ.get("BATCH", "32"))
CHUNK = int(os.environ.get("CHUNK", "20"))         # action chunk length (2 s at 10 Hz)
N_ACTION_STEPS = int(os.environ.get("N_ACTION_STEPS", "10"))
SEED = int(os.environ.get("SEED", "1000"))
OUT = f"outputs/train/{MODEL.split('/')[-1]}"

# %%
# Install (Colab: ~2 min). Pin to the same version as the repo.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "lerobot[dataset,training]==0.6.1"], check=True)

# %%
# Hugging Face login from Colab secrets.
try:
    from google.colab import userdata  # type: ignore
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception:
    pass
from huggingface_hub import login, whoami
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("HF user:", whoami()["name"])

# %%
# Train. `lerobot-train` writes checkpoints under OUT/checkpoints/<step>/pretrained_model and pushes the last one.
cmd = [
    "lerobot-train",
    f"--dataset.repo_id={DATASET}",
    "--policy.type=act",
    f"--policy.chunk_size={CHUNK}",
    f"--policy.n_action_steps={N_ACTION_STEPS}",
    "--policy.device=cuda",
    "--policy.push_to_hub=true",
    f"--policy.repo_id={MODEL}",
    "--policy.private=true",
    f"--output_dir={OUT}",
    f"--job_name={MODEL.split('/')[-1]}",
    f"--steps={STEPS}",
    f"--batch_size={BATCH}",
    f"--seed={SEED}",
    "--save_freq=5000",
    "--log_freq=100",
    "--wandb.enable=false",
    "--num_workers=4",
]
print(" ".join(cmd))
subprocess.run(cmd, check=True)

# %%
# Keep the training log next to the model (the report needs the loss curve).
from huggingface_hub import HfApi
api = HfApi()
for name in ("train.log", "logs"):
    p = os.path.join(OUT, name)
    if os.path.exists(p):
        api.upload_folder(repo_id=MODEL, folder_path=p, path_in_repo=f"train_logs/{name}") if os.path.isdir(p) else \
            api.upload_file(repo_id=MODEL, path_or_fileobj=p, path_in_repo=f"train_logs/{name}")
print("done:", MODEL)
