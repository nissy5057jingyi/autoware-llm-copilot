"""
保存所有场景所有候选的 BEV 图像
"""
import sys, os, base64
sys.path.insert(0, os.path.expanduser('~/autonomous_driving/DAPO'))
from build_preference_vldpo import render_bev
from eval_vldpo_vs_numeric import make_scenes
from pathlib import Path

out_dir = Path(os.path.expanduser("~/autonomous_driving/DAPO/bev_images"))
out_dir.mkdir(exist_ok=True)

scenes = make_scenes()

for scene in scenes:
    scene_dir = out_dir / scene["id"]
    scene_dir.mkdir(exist_ok=True)

    # GT 轨迹
    gt_b64 = render_bev(scene["gt_trajectory"], title="GT trajectory")
    with open(scene_dir / "GT.png", "wb") as f:
        f.write(base64.b64decode(gt_b64))

    # 每个候选
    valid = [c for c in scene["candidates"] if c["parse"]["parse_ok"]]
    for c in valid:
        traj = c["trajectory"]
        action_str = "_".join([f"{a[0][:3]}{a[1][:3]}" for a in c["actions"]])
        title = f"Rank{c['rank']} r={c['reward']['total']:.2f} {action_str}"
        b64 = render_bev(traj, candidate_idx=c["rank"]-1, title=title)
        fname = scene_dir / f"rank{c['rank']}_r{c['reward']['total']:.2f}.png"
        with open(fname, "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"  saved: {fname.name}")

    print(f"[{scene['id']}] {len(valid)+1} images saved to {scene_dir}")

print(f"\n✓ All BEV images saved to: {out_dir}")
print(f"  Total files: {len(list(out_dir.rglob('*.png')))}")
