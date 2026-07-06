# setup_ssh_passwordless.sh

给局域网远端机器配置 SSH 免密登录。

```bash
./setup_ssh_passwordless.sh -H REMOTE_IP -u zc
./setup_ssh_passwordless.sh -H zc@REMOTE_IP:2222
```

含义：`REMOTE_IP` 是远端机器 IP，`-u zc` 是远端用户名；第二条表示远端 SSH 端口是 `2222`。

把 `REMOTE_IP` 换成目标机器的局域网 IP。首次运行会要求输入远端用户密码；成功后可直接 `ssh zc@REMOTE_IP` 登录。
