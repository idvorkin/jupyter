# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + tags=[]
# Open CV
# -

from matplotlib import pyplot as plt
import cv2
from IPython.display import display, Image, clear_output
import os


# %run "~/rare_gits/video-edit/cv_helper.py"

# %run "~/rare_gits/video-edit/filter-to-motion.py"

class remove_background:
    def __init__(self, base_filename, in_fps=30):
        self.base_filename = base_filename
        self.in_fps = in_fps
        self.debug_window_refresh_rate = int(
            self.in_fps / 2
        )  # every 0.5 seconds; TODO Compute
        pass

    def create(self, input_video):
        self.state = FrameState(0, 0)

    def destroy(self):
        pass

    def frame(self, idx, original_frame):
        self.state.idx = idx

        # PERF: Processing at 1/4 size boosts FPS by TK%
        in_frame = shrink_image_half(original_frame)

        # PERF: Motion Mask sampled frames
        motion_mask = to_motion_mask_fast(self.state, in_frame)

        # skip frames with no motion
        if is_frame_black(motion_mask):
            return

        # PERF - show_debug_window at on sampled frames
        if idx % self.debug_window_refresh_rate == 0:
            debug_frame = create_analyze_debug_frame(in_frame, motion_mask)
            burn_in_debug_info(debug_frame, idx, self.in_fps)
            masked_input = cv2.bitwise_and(in_frame, in_frame, mask=motion_mask)
            cv_helper.display_jupyter(masked_input)


# +
input_video_path = os.path.expanduser("~/downloads/igor-magic.mp4")
input_video = cv2.VideoCapture(input_video_path)
if not input_video.isOpened():
    print(f"Unable to Open {input_video_path}")
    1 / 0

ic(f"Processing File {input_video_path}")
rb = remove_background(input_video_path, 30)
cv_helper.process_video(input_video, rb)
# -




