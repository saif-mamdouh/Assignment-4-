from huggingface_hub import HfApi
from pathlib import Path
import requests




def download_checkpoints_from_hf(repo_id: str, local_dir: str):
"""Download all files under `checkpoints/` from a model repo on Hugging Face.
Requires HF token to be logged in via huggingface-cli.
"""
api = HfApi()
files = api.list_repo_files(repo_id)
local_dir = Path(local_dir)
local_dir.mkdir(parents=True, exist_ok=True)
for f in files:
if f.startswith('checkpoints/') and (f.endswith('.pt') or f.endswith('.pth')):
dest = local_dir / Path(f).name
if dest.exists():
print('Skipping existing', dest)
continue
print('Downloading', f)
try:
api.download_repo_file(repo_id=repo_id, path_in_repo=f, local_dir=str(local_dir))
except Exception as e:
print('Failed to download', f, e)