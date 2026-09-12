# lan_share_mount.sh

在主机器上一键共享本地目录，并让远端机器挂载后可读写。

```bash
./lan_share_mount.sh
./lan_share_mount.sh -H zc@REMOTE_IP -r /mnt/main-share
./lan_share_mount.sh -H zc@REMOTE_IP -r '~/main-share'
```

含义：`zc@REMOTE_IP` 是远端机器；`/mnt/main-share` 是远端挂载目录；本机是运行脚本的主机器。

把 `REMOTE_IP` 换成目标机器的局域网 IP。远端路径里的 `~` 要加引号，否则会先被本机展开。默认共享本机 `/run/timeshift/backup/share`。加 `--persist` 可让远端开机自动挂载。

注意：NFS 会暴露真实目录树；若共享目录含 `.codex/`，脚本会默认提醒并阻止，确认要共享时加 `--allow-codex`。
目标 host key 必须已独立核对并记录在本机 `known_hosts` 中；挂载流程不会自动接受
未知或变化的密钥。
