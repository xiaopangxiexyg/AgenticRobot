# Wrist Camera Robot-Body Mask Painter

This tool opens a saved wrist-camera video, lets you pick one frame, paint the
robot body/self area, and saves a binary PNG mask for that painted area.

Run from the repository root:

```bash
python utils/wrist_cam_mask/painting/paint_video_mask.py /path/to/episode_000000.mp4
```

If the video path is omitted, the tool opens a file picker:

```bash
python utils/wrist_cam_mask/painting/paint_video_mask.py
```

Controls:

- The window shows original image on the left and painted preview on the right.
- Images resize with the window while preserving aspect ratio.
- `帧` slider: choose the frame to annotate.
- `涂抹` / `擦除` radio buttons: choose whether left mouse drag paints or erases.
- Right mouse drag: erase directly.
- Buttons: `保存 Mask`, `撤回`, `清空`, `反选`, `选择视频`, `退出`.
- `笔刷` controls: use `-` / `+`, the slider, or `[` / `]` to change brush radius.
- `透明度` slider: change the painted preview opacity.
- `保存源帧` checkbox: also save the selected frame image next to the mask.
- `p` / `e`: switch paint / erase mode.
- `u` or `z`: undo last stroke.
- `c`: clear mask.
- `i`: invert mask.
- `s`: save mask PNG and metadata JSON.
- `q` or `Esc`: quit.

Outputs default to:

- GUI / masks: `utils/wrist_cam_mask/painting/`
- Metadata configs: `utils/wrist_cam_mask/configs/`

The PNG mask is `uint8`: `255` means the painted robot body/self region, and
`0` means background. Use `--save-frame` if you also want to save the selected
source frame next to the mask for checking.
