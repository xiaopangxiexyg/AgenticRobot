# Replay 参数说明

Replay 重放已有数据；运行命令见 [real README](./README.md)。离线浏览与真机执行使用同一
GUI 入口，但只有显式提供真实 backend 并经过人工放行才允许运动。

## 1. Replay GUI：离线浏览（R1）

可直接复制的命令见 [real README 的第 2 节](./README.md)。浏览器访问
`http://127.0.0.1:8765/`。默认只允许 loopback；不要为方便而常开
`--unsafe-allow-remote-web`。

完整 GUI、token、视频/触觉显示说明见
[`teleop_runtime/replay/README.md`](../../teleop_runtime/replay/README.md)。

## 2. 真机 Replay（M）

真机 replay 仍使用同一个入口，但必须显式提供 active sides、RM65 IP、左右 tool profile
以及需要的 end-effector backend。例如 DMTAC 配置骨架：

```bash
python -m teleop_runtime.replay \
  --dataset-root "${DATASET_ROOT:?set DATASET_ROOT}" \
  --active-sides both \
  --left-arm-ip "${LEFT_ARM_IP:?set LEFT_ARM_IP}" \
  --right-arm-ip "${RIGHT_ARM_IP:?set RIGHT_ARM_IP}" \
  --left-tool-profile hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json \
  --right-tool-profile hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json \
  --end-effectors-enabled \
  --hand-backend dmtac-gripper \
  --dmtac-config latest \
  --require-dmtac-config \
  --host 127.0.0.1 \
  --port 8765 \
  --no-open-browser
```

三个必需环境变量未设置时 Bash 会直接停止。启动前还要人工检查所选 episode 的 control mode
与当前 backend 一致；这个骨架不会替你判断轨迹是否适合当前工作空间。

启动服务不等于批准运动。先在 GUI 检查 dataset/episode/control mode，执行 `Move Start`
安全初始化，确认姿态后再人工 `Start`。Position replay 使用数据集 main action，不重复 live
teleop 的 pre-action 平滑；主 action 之后的 IK、安全限位和下发仍保留。

MuJoCo replay 不属于本页；canonical 与 direct-qpos debug 的边界统一放在
[仿真说明](../sim/MUJOCO.md)。
