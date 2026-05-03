# portable_vis

独立版可视化目录。运行时只依赖本目录中的代码和资产，以及外部输入数据集本身。

## 用法

```bash
python /home/zc/HandMotion/UV_conversion/visualization/portable_vis/quick_sample_visualizations.py /path/to/dataset_root --sample-count 3 --output_dir /tmp/spot_checks
python /home/zc/HandMotion/UV_conversion/visualization/portable_vis/visualize_lerobot_tactile.py /path/to/dataset_root --episode 12 --output_dir /tmp/tactile_vis
python /home/zc/HandMotion/UV_conversion/visualization/portable_vis/visualize_lerobot_mano.py /path/to/dataset_root --episode 12 --output_dir /tmp/mano_vis
python /home/zc/HandMotion/UV_conversion/visualization/portable_vis/visualize_lerobot_mano.py /path/to/dataset_root --episode 12 --camera_intrinsics_json /path/to/camera_intrinsics.json --output_dir /tmp/mano_vis
python /home/zc/HandMotion/UV_conversion/visualization/portable_vis/visualize_lerobot_spatial_tactile.py /path/to/dataset_root --episode 12 --render-mode both --output_dir /tmp/spatial_tactile_vis
```

常规依赖: `python`, `numpy`, `opencv-python`, `pyarrow`, `torch`, `scipy`, `tqdm`。

`visualize_lerobot_mano.py` 会优先使用显式 `--camera_intrinsics_json`，其次使用 dataset 内的 `meta/camera_intrinsics*.json`，再尝试原始 `session_dir/meta.json`，最后回退到 `assets/camera_intrinsics/` 里的便携内参资产。

`visualize_lerobot_spatial_tactile.py` 现在支持 `--render-mode tactile|mano|both`。其中 `both` 会同时叠加 MANO 网格和 spatial tactile 点云；旧参数 `--with-mano` 仍兼容，等价于 `--render-mode both`。

## 代码

- `quick_sample_visualizations.py`
- `visualize_lerobot_mano.py`
- `visualize_lerobot_tactile.py`
- `common/`
- `manotorch/`
- `mano/webuser/`

## 文件资产

- `assets/camera_intrinsics/`
- `assets/mano/models/MANO_LEFT.pkl`
- `assets/mano/models/MANO_RIGHT.pkl`
