"""Run harmonization with frame-matched walking BVH for all 6 models."""
import subprocess, sys, os

SINMDM = r"D:\sinMDM\sinmdm"
MODELS = ["01-2-FeiTian","02-1-PuSa","03-2-LianHuaTongZi","04-1-LiShiWuJi","05-2-JiGuJiYue","06-2-PiPaJiYue"]

env = os.environ.copy()
env["PYTHONPATH"] = SINMDM + os.pathsep + env.get("PYTHONPATH", "")

for name in MODELS:
    model_path = os.path.join(SINMDM, "save", name, "model000014999.pt")
    walking_bvh = os.path.join(SINMDM, "dataset", f"walking_{name}.bvh")
    output_dir = os.path.join(SINMDM, "save", name, "harmonization_walking")
    
    print(f"\n>>> {name}")
    cmd = [
        sys.executable, os.path.join(SINMDM, "sample", "edit.py"),
        "--model_path", model_path,
        "--edit_mode", "harmonization",
        "--ref_motion", walking_bvh,
        "--num_samples", "2",
        "--batch_size", "2",
        "--output_dir", output_dir,
        "--device", "0",
    ]
    
    result = subprocess.run(cmd, cwd=SINMDM, env=env, capture_output=True, text=True, timeout=600)
    if result.returncode == 0:
        bvhs = [f for f in os.listdir(output_dir) if f.endswith('.bvh')]
        print(f"  OK: {len(bvhs)} BVH files")
    else:
        err = result.stderr.strip().split('\n')
        print(f"  FAIL: {err[-1] if err else 'unknown'}")

print("\nDone!")
