---
title: Linux
---

# Linux初始化

```sh
sudo apt update -y
sudo apt upgrade -y

sudo apt install tmux zsh htop git lsof vim curl wget zsh -y
chsh -s /bin/zsh
```

## zsh

国内版：

```sh
git clone https://gitee.com/mirrors/oh-my-zsh.git ~/.oh-my-zsh
cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc
sh -c "$(curl -fsSL https://gitee.com/jsharkc/jacobin-zsh-theme/raw/master/install.sh)"
git clone https://gitee.com/jsharkc/zsh-autosuggestions.git $ZSH_CUSTOM/plugins/zsh-autosuggestions
git clone https://gitee.com/jsharkc/zsh-syntax-highlighting.git $ZSH_CUSTOM/plugins/zsh-syntax-highlighting

source ~/.zshrc
```

国际版：

```sh
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
sh -c "$(curl -fsSL https://raw.githubusercontent.com/Jsharkc/jacobin-zsh-theme/master/install.sh)" 
git clone https://github.com/zsh-users/zsh-autosuggestions.git $ZSH_CUSTOM/plugins/zsh-autosuggestions
git clone https://github.com/zsh-users/zsh-syntax-highlighting.git $ZSH_CUSTOM/plugins/zsh-syntax-highlighting

source ~/.zshrc
```

编辑 `~/.zshrc`：

```sh
plugins=(git zsh-autosuggestions zsh-syntax-highlighting)
```

## Python

uv：https://zhuanlan.zhihu.com/p/1894040611625084682

uv 镜像：
```bash
UV_PYTHON_INSTALL_MIRROR=https://github.com/indygreg/python-build-standalone/releases/download uv python install 3.13.1
export UV_DEFAULT_INDEX="https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
```

pip 镜像：
- https://mirrors.tuna.tsinghua.edu.cn/help/pypi/
- https://mirror.tuna.tsinghua.edu.cn/help/anaconda/

清理缓存：

```sh
conda clean -p -y
conda clean -t -y
conda clean --all -y
rm -rf ~/.cache/pip
# C:\Users\username\AppData\Local\pip\cache
```

PyTorch: [Previous PyTorch Versions | PyTorch](https://pytorch.org/get-started/previous-versions/)

Check torch version:

```
python -m torch.utils.collect_env
```

Check torch GPU:

```python
import torch

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
else:
    print("No GPUs available. PyTorch will run on CPU.")
```

Show real-time GPU memory:

```
watch -n 1 nvidia-smi
```

## ssh

配置密钥：

```sh
vim ~/.ssh/authorized_keys
echo SOME_KEY >> ~/.ssh/authorized_keys
```

生成ssh key：

```sh
git config --global user.name "yorhaha"
git config --global user.email "yorhaha@outlook.com"
ssh-keygen -t rsa -C "yorhaha@outlook.com"
cat ~/.ssh/id_rsa.pub
```

重新生成某个ip的key：

```sh
ssh-keygen -R ip
```

## tmux

`vim ~/.tmux.conf`

```
unbind C-b
set -g prefix C-a
bind C-a send-prefix

bind -n M-Left select-pane -L
bind -n M-Right select-pane -R
bind -n M-Up select-pane -U
bind -n M-Down select-pane -D

set -g base-index 1
setw -g pane-base-index 1
setw -g automatic-rename on

set -g history-limit 10000
set -g default-terminal "screen-256color"
set -g mouse on
```

## GPU

nvitop: https://github.com/XuehaiPan/nvitop

检查 PyTorch 配置：

```sh
python -m torch.utils.collect_env
```

检查能否调用GPU：

```python
import torch

if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"GPU {i}: {gpu_name}")
else:
    print("No GPUs available. PyTorch will run on CPU.")
```

## Proxy

```
export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
export all_proxy="socks5://127.0.0.1:7890"

export http_proxy=""
export https_proxy=""
export all_proxy=""

git config --global http.proxy http://127.0.0.1:7890
git config --global https.proxy http://127.0.0.1:7890

git config --global --unset http.proxy
git config --global --unset https.proxy
```

端口转发：https://www.cnblogs.com/cangqinglang/p/12732661.html

将本地端口映射到服务器：

```sh
ssh -R 7890:localhost:7890 username@server_ip
```

将服务器端口映射到本地：

```sh
ssh -NL 9999:127.0.0.1:9999 webarena@36.102.215.18 -p 2202
```

## LLM

modelscope：https://www.modelscope.cn/models

```
pip install modelscope
modelscope download --model $model_name --local_dir $save_dir
```

HuggingFace：https://hf-mirror.com/

```
token=
user_name=
model_id=Qwen/Qwen3-32B
save_dir=Qwen3-32B

huggingface-cli download --resume-download $model_id --local-dir $save_dir --local-dir-use-symlinks False --token $token

sudo apt-get install aria2
# hfd <repo_id> [--include include_pattern] [--exclude exclude_pattern] [--hf_username username] [--hf_token token] [--tool aria2c|wget] [-x threads] [--dataset] [--local-dir path]
./hfd.sh $model_id --tool aria2c -x 4 --hf_username $user_name --hf_token $token

wget --header="Authorization: Bearer ${token}" HF_FILE_URL

curl -L --header "Authorization: Bearer ${token}" -o model-00001-of-00030.safetensors HF_FILE_URL
```

## 文件传输

```
rsync -avz --partial --progress source_file username@ip:target_dir
```
