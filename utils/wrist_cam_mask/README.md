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

To continue editing an existing mask, load its saved metadata JSON:

```bash
python utils/wrist_cam_mask/painting/paint_video_mask.py --load utils/wrist_cam_mask/configs/right_dmtac_m_mask.json
```

You can also open the GUI first and click `加载 Mask/配置`, then choose either
the saved JSON in `utils/wrist_cam_mask/configs/` or the mask PNG in
`utils/wrist_cam_mask/painting/`.

Controls:

- The window shows original image on the left and painted preview on the right.
- Images resize with the window while preserving aspect ratio.
- `帧` slider: choose the frame to annotate.
- `涂抹` / `擦除` radio buttons: choose whether left mouse drag paints or erases.
- Right mouse drag: erase directly.
- Buttons: `保存 Mask`, `撤回`, `清空`, `反选`, `选择视频`, `退出`.
- `加载 Mask/配置`: load an existing mask PNG or metadata JSON to continue painting.
- `保存文件名(留空自动)`: set a custom output stem, for example `right_dmtac_m_mask`.
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
- Paths written inside metadata JSON are relative to the repository root.

Custom output names can also be passed from the command line:

```bash
python utils/wrist_cam_mask/painting/paint_video_mask.py /path/to/video.mp4 --output-name right_dmtac_m_mask
```

This writes `utils/wrist_cam_mask/painting/right_dmtac_m_mask.png` and
`utils/wrist_cam_mask/configs/right_dmtac_m_mask.json` unless those names
already exist, in which case `_01`, `_02`, ... is appended.

The PNG mask is `uint8`: `255` means the painted robot body/self region, and
`0` means background. Use `--save-frame` if you also want to save the selected
source frame next to the mask for checking.
