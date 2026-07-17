# 生成 UMI→desk 标定
python -m utils.umi_hand_eye --side left

# 不初始化设备，只检查配置、标定有效期和 v2.3 layout
python -m teleop_runtime.collection umi \
  --config teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json \
  --validate-only

# 正式启动
python -m teleop_runtime.collection umi \
  --config teleop_runtime/configs/umi/dual_wrist_dmtac_v1.json

# 调试时禁止真实夹爪指令
python -m teleop_runtime.collection umi --no-send-gripper

# 启用现有四踏板输入
python -m teleop_runtime.collection umi --foot-pad