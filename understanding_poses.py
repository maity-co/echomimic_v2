import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from src.utils.dwpose_util import draw_pose_select_v2


for index in range(0, 240):
    tgt_musk = np.zeros((768, 768, 3), dtype='uint8')
    tgt_musk_path = os.path.join('./assets/halfbody_demo/pose/01', f"{index}.npy")
    detected_pose = np.load(tgt_musk_path, allow_pickle=True).tolist()
    imh_new, imw_new, rb, re, cb, ce = detected_pose['draw_pose_params']

    im = draw_pose_select_v2(detected_pose, imh_new, imw_new, ref_w=800)
    im = np.transpose(np.array(im), (1, 2, 0))
    tgt_musk[rb:re, cb:ce, :] = im

    tgt_musk_pil = Image.fromarray(tgt_musk).convert('RGB')

    # Display the image interactively
    plt.imshow(tgt_musk_pil)
    plt.title(f'Image {index}')
    plt.axis('off')
    plt.show()  # waits until the window is closed

