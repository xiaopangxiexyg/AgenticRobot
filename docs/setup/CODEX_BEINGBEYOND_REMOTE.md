# 远端 Codex 使用 BeingBeyond Provider

> 文档类型：特定远端开发工具运维说明。本文不定义 DexTeleop 的 Conda/Python 依赖、
> policy deployment、远端硬件控制或项目运行合同。示例中的 host、shell、provider 名称和
> endpoint 均需在目标机器重新确认；密钥不得写入仓库、命令历史、日志或聊天。

本文说明在 VS Code Remote 主机上使用 BeingBeyond provider 和其专用 API key 的标准配置、
验证闭环与故障分支。`dex-remote` 和 zsh 是原始排障现场的示例，不是所有开发机器的默认值。

## 1. 正确配置逻辑

数据流应为：

```text
BeingBeyond key
  -> 远端 OPENAI_API_KEY 环境变量
  -> VS Code Server 启动环境
  -> Codex app-server
  -> config.toml 中 env_key = "OPENAI_API_KEY"
  -> BeingBeyond endpoint
```

关键规则：

- `env_key` 填的是**环境变量名**，不能填密钥值。
- BeingBeyond 专用密钥使用环境变量认证：`requires_openai_auth = false`。
- `~/.codex/auth.json` 不是该自定义 provider 的密钥来源，不要手工修改它。
- VS Code/Codex 进程只会继承其启动时的环境；在旧终端里 `export` 不会更新已运行进程。
- 密钥只保存在权限为 `0600` 的独立文件中，不写入仓库、`config.toml`、命令历史或聊天。

OpenAI 官方语义见 [Authentication](https://learn.chatgpt.com/docs/auth) 和
[Config Reference](https://learn.chatgpt.com/docs/config-file/config-reference)。

## 2. 标准修复步骤

### 2.1 先轮换泄露或误配过的密钥

如果错误日志曾显示一段密钥作为“缺失的环境变量名”，说明密钥曾被误填进 `env_key`，
并已进入日志。先在 BeingBeyond 后台撤销旧密钥并创建新密钥；不要通过删除日志来代替轮换。

### 2.2 检查 `~/.codex/config.toml`

保留实际 provider 名称、模型和 endpoint；认证部分应满足：

```toml
model_provider = "beingbeyond"

[model_providers.beingbeyond]
name = "beingbeyond"
base_url = "https://<beingbeyond-endpoint>/v1"
env_key = "OPENAI_API_KEY"
wire_api = "responses"
requires_openai_auth = false
```

不要写成：

```toml
env_key = "<实际密钥>"
```

若服务支持 HTTPS，应使用 HTTPS。使用 HTTP 会使 bearer key 缺少传输层加密保护。

### 2.3 创建独立密钥文件

在远端终端执行：

```bash
install -d -m 700 "$HOME/.config/codex"
install -m 600 /dev/null "$HOME/.config/codex/beingbeyond.env"
nano "$HOME/.config/codex/beingbeyond.env"
```

文件只包含一行；在远端编辑器中直接填写新密钥：

```bash
export OPENAI_API_KEY='<新的 BeingBeyond key>'
```

不要把真实密钥作为 shell 命令参数粘贴，以免进入 history 或进程参数。

### 2.4 从正确的 shell 启动文件加载

先确认登录 shell：

```bash
getent passwd "$USER" | cut -d: -f7
```

当前 `dex-remote` 使用 `/bin/zsh`。编辑 `~/.zshenv`：

```bash
nano "$HOME/.zshenv"
```

加入：

```bash
if [ -r "$HOME/.config/codex/beingbeyond.env" ]; then
    . "$HOME/.config/codex/beingbeyond.env"
fi
```

然后设置普通启动文件权限：

```bash
chmod 644 "$HOME/.zshenv"
```

这里使用 `~/.zshenv`，因为 zsh 的非交互进程也需要环境变量。只写 `~/.bashrc` 对 zsh
无效；只写 `~/.zshrc` 可以修复交互终端，但不能可靠覆盖 VS Code Server 的启动链路。

如果上一条命令输出的是 `/bin/bash`，按下一节配置 Bash；不要把 Bash 和 zsh 的启动文件混用。

### 2.5 如果远端使用 Bash

Bash 的启动文件规则与 zsh 不同：

- 交互、非登录 Bash 通常读取 `~/.bashrc`。
- 登录 Bash 按 `~/.bash_profile`、`~/.bash_login`、`~/.profile` 的顺序读取第一个存在的文件，
  不会自动读取 `~/.bashrc`。
- 因此，加载语句必须放进 VS Code Server 启动时实际经过的文件。常见做法是在 `~/.bashrc`
  中放置加载语句，并让实际使用的 Bash 登录文件显式加载 `~/.bashrc`；不要在多个文件中重复添加。

编辑 `~/.bashrc`：

```bash
nano "$HOME/.bashrc"
```

加入与 zsh 相同的加载语句：

```bash
if [ -r "$HOME/.config/codex/beingbeyond.env" ]; then
    . "$HOME/.config/codex/beingbeyond.env"
fi
```

然后设置普通启动文件权限：

```bash
chmod 644 "$HOME/.bashrc"
```

如果实际使用的登录文件是 `~/.bash_profile`、`~/.bash_login` 或 `~/.profile`，确认其中已有
等效的 `source "$HOME/.bashrc"`，或者把上面的加载语句直接放进该登录文件。Bash 非交互进程
不会自动读取 `~/.bashrc`；如果 VS Code Server 是由服务管理器直接启动，必须在该服务的父环境
或 `EnvironmentFile` 中注入 `OPENAI_API_KEY`，不能仅依赖 `~/.bashrc`。

### 2.6 验证新 shell，不回显密钥

zsh 使用：

```bash
source "$HOME/.zshenv"
```

Bash 使用：

```bash
source "$HOME/.bashrc"
```

如果加载语句放在 Bash 登录文件中，则改为 `source` 实际使用的那个文件。随后执行共同的存在性检查：

```bash
if [ -n "${OPENAI_API_KEY:-}" ]; then echo "OPENAI_API_KEY present"; else echo "OPENAI_API_KEY missing"; fi
```

必须输出 `OPENAI_API_KEY present`。不要运行 `echo "$OPENAI_API_KEY"`。

### 2.7 完全重启 VS Code Remote

1. 关闭所有连接该远端的 VS Code 窗口。
2. 在 VS Code 命令面板运行 `Remote-SSH: Kill VS Code Server on Host...`。
3. 选择 `dex-remote`，等待远端 VS Code Server 退出。
4. 重新连接远端并新建终端，再执行上一节的存在性检查。
5. 新建一次 Codex 对话并发出最小请求，确认不再出现 `missing_environment_variable`。

`Reload Window`、新开终端或 `source` 启动文件都不会改变已经运行的 Codex app-server；
必须重启整个 VS Code Server。

## 3. 故障分支

### A. 新终端仍显示 `OPENAI_API_KEY missing`

依次检查：

```bash
test -r "$HOME/.config/codex/beingbeyond.env" && echo "env file readable" || echo "env file unreadable"
```

```bash
grep -n 'beingbeyond.env' \
    "$HOME/.zshenv" "$HOME/.zshrc" \
    "$HOME/.bash_profile" "$HOME/.bash_login" "$HOME/.profile" "$HOME/.bashrc" 2>/dev/null
```

- 登录 shell 是 zsh：加载语句必须进入 `~/.zshenv`。
- 登录 shell 是 Bash：加载语句必须进入实际使用的 Bash 启动文件，或由该文件加载 `~/.bashrc`。
- 环境文件应归当前用户所有、权限为 `0600`。
- 文件中的变量名必须精确为 `OPENAI_API_KEY`。
- 修正后重新 `source` 对应的启动文件，先让当前终端通过存在性检查。

### B. 终端显示 `present`，Codex 仍报缺少变量

这是旧 VS Code Server/Codex app-server 仍在运行：

1. 关闭该主机的全部 VS Code 窗口。
2. 执行 `Remote-SSH: Kill VS Code Server on Host...`。
3. 重新连接后再试。

从本地 DexTeleop 仓库可只读确认旧 Codex 进程是否仍存在：

```bash
./codex_remote/remote_ps.sh -H dex-remote --grep codex
```

不要用宽泛的 `pkill`，避免停止其他用户会话或无关服务。

### C. 错误把一长段密钥当作“缺失的环境变量名”

原因是 `env_key` 被填成了真实密钥。立即执行：

1. 把 `env_key` 改回 `"OPENAI_API_KEY"`。
2. 在 BeingBeyond 后台轮换该密钥。
3. 更新权限为 `0600` 的环境文件。
4. 完全重启 VS Code Server。

### D. 不再报 missing，但出现 `401`、`403` 或 `unauthorized`

环境注入已经成功，问题转到服务端认证：

- 确认密钥未过期、未撤销，并有目标模型权限。
- 确认 `base_url` 指向 BeingBeyond 的正确 API 根路径。
- 确认模型名确实由该 endpoint 提供。
- 不要反复把同一密钥写入 `auth.json` 或 `env_key`。

### E. 出现 DNS、连接超时、TLS 或证书错误

这不是密钥注入错误。检查远端到 BeingBeyond endpoint 的网络、DNS、代理和证书链。
优先使用服务方提供的 HTTPS endpoint；不要为绕过证书错误而关闭 TLS 校验。

### F. 出现 `Remote Control requires ChatGPT authentication`

这是 Codex Remote Control 的独立限制。BeingBeyond provider 的普通模型请求可以继续使用
provider key；不要为消除此提示把 `requires_openai_auth` 改成 `true`。只有确实需要
ChatGPT Remote Control 时，才单独配置 ChatGPT 登录。

### G. SSH 终端提示 `codex: command not found`

VS Code 扩展可以使用扩展内置的 Codex 二进制，SSH 的 `PATH` 中没有独立 `codex` 命令
并不会造成 `missing_environment_variable`。不要把 CLI 安装问题和 provider 认证混为一谈。

## 4. 最终验收

修复完成必须同时满足：

- `config.toml` 中 `env_key = "OPENAI_API_KEY"`、`requires_openai_auth = false`。
- 密钥文件存在、归当前用户所有且权限为 `0600`。
- zsh 的 `~/.zshenv` 或 Bash 的实际启动文件加载密钥文件，不输出任何内容。
- 新远端终端显示 `OPENAI_API_KEY present`。
- VS Code Server 已完全重启。
- 新 Codex 请求不再出现 `missing_environment_variable`。
- 若出现 `401/403`，按服务端密钥/权限分支处理，而不是继续修改 shell。

## 5. 回退

若需要停用 BeingBeyond：

1. 注释或删除对应 shell 启动文件中加载 `beingbeyond.env` 的三行。
2. 在当前终端执行 `unset OPENAI_API_KEY`。
3. 完全重启 VS Code Server。
4. 在 BeingBeyond 后台撤销不再使用的密钥。

不要提交、同步或分享 `~/.config/codex/beingbeyond.env`。
