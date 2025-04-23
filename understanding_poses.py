import numpy as np
import cv2
import os
from tqdm import tqdm
from src.utils.dwpose_util import draw_pose_select_v2

# === Setup Paths ===
pose_dir = './assets/halfbody_demo/from_vids/02'
output_path = "./output_pose_video.mp4"
frame_size = (768, 768)
fps = 25

# === Get List of NPY Files ===
npy_files = sorted(
    [f for f in os.listdir(pose_dir) if f.endswith('.npy')],
    key=lambda x: int(os.path.splitext(x)[0])  # sort by numeric index
)

# === Setup Video Writer ===
video_writer = cv2.VideoWriter(
    output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size
)

# === Process Frames ===
for fname in tqdm(npy_files, desc="Generating video from poses"):
    full_path = os.path.join(pose_dir, fname)
    detected_pose = np.load(full_path, allow_pickle=True).tolist()

    tgt_musk = np.zeros((768, 768, 3), dtype='uint8')
    imh_new, imw_new, rb, re, cb, ce = detected_pose['draw_pose_params']

    im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=800)
    im = np.transpose(np.array(im), (1, 2, 0))  # CHW -> HWC
    tgt_musk[rb:re, cb:ce, :] = im

    frame_bgr = cv2.cvtColor(tgt_musk, cv2.COLOR_RGB2BGR)
    video_writer.write(frame_bgr)

# === Finalize ===
video_writer.release()
print(f"✅ Video saved to {output_path}")
