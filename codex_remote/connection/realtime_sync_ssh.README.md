# realtime_sync_ssh.sh

把本机目录实时单向同步到远端目录：本机 -> 远端。

```bash
./realtime_sync_ssh.sh -H REMOTE_IP -u zc -l ~/Release -r '~/Release'
```

含义：`-H REMOTE_IP -u zc` 是远端机器；`-l ~/Release` 是本机目录；`-r '~/Release'` 是远端目录。

注意：远端路径里的 `~` 要加引号，否则会先被本机展开。

默认不保留本机属主/属组，避免远端普通用户因 `chgrp` 权限失败中断。

默认跳过 `.codex/` 目录。如果本地同步根目录存在
`.realtime_sync_ssh.exclude`，脚本还会自动加载其中的非空、非注释规则；
设备分配、标定、数据集和其他 machine-local 路径应统一记录在该文件中。

默认保留远端新增的独有文件/目录；只有显式加 `--delete` 才会删除远端多出来的内容。

如果本机 inotify 限制太低，脚本会提示提高 `max_user_watches/max_user_instances`，否则 `lsyncd` 可能无法启动。

如果 SSH 还未免密，会自动调用 `setup_ssh_passwordless.sh`。按 `Ctrl-C` 停止实时同步。
