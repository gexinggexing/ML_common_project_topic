ssh配置查询： "C:\Users\Lenovo\.ssh\config"
Jhf@171821

1. Lenovo 本机    ⭐【新建python环境优先建在 】
连接方式：本地直接使用
GPU：1 张 RTX 5080
显存：16 GB
用途：本地调试、小模型测试、数据处理

2. hs / HuoShan2 服务器  ⭐【新建python环境优先建在 D盘】
连接方式：ssh hs
"C:\Users\Lenovo\.ssh\id_ed25519_oldserver.pub"
资源情况：
Host：di-20260417182420-kz6q9
系统：Ubuntu 24.04.2 LTS
GPU：2 张 NVIDIA A800-SXM4-80GB
显存：每张 80 GB
CPU：Intel Xeon Platinum 8336C
CPU 核心：128 CPUs
内存：463 GB
主要存储：
/vePFS-0x0d：6.0 TB，总体已用 74%
/：98 GB，已用 89%
数据盘：
/mnt/dataset0-4：未挂载
/data：不存在
用途建议：
最适合跑重任务
适合大模型训练、fMRI / EEG foundation model、MindEye2、Brain-Semantoks、LCM 等任务
需要注意代码和数据主要放在 /vePFS-0x0d/nzh⚠️（除了这个nzh目录下以外其他地方的文件没有我的允许不能做任何修改，只能看！！！）

3. l40 / amax 服务器  ⭐【新建python环境优先建在：/data/ningzh/envs；按理说/home/ningzh下面也是放环境的地方，但是奈何这里已经满了，迫不得已再考虑把环境建在这里】
连接方式：ssh l40
"C:\Users\Lenovo\.ssh\id_ed25519_oldserver.pub"
资源情况：
Host：amax
系统：Ubuntu 20.04.6 LTS
GPU：8 张 NVIDIA L40
显存：每张约 46 GB
CPU：AMD EPYC 9554 64-Core Processor
CPU 核心：128 CPUs
内存：503 GB
主要存储：
/data：15 TB，总体已用 70%    
/home：3.3 TB，但是已满，100%
/mnt/dataset0：143 TB，已用 91%
/mnt/dataset1：100 TB，已满，100%
/mnt/dataset2：53 TB，已用 86%
/mnt/dataset3：158 TB，已用 78%
/mnt/dataset4：415 TB，已用 97%
环境情况：
conda 可用
常用环境包括：
/data/ningzh/LCM/envs/lcm
/data/ningzh/envs/bsem
/data/ningzh/envs/pt24-cu121-py312
/home/ningzh/miniconda3/envs/general
当前 base 环境没有 torch
nvcc 版本：CUDA 10.1
nvidia-smi 显示驱动支持 CUDA 12.2
当前 GPU 状态：
查询时 8 张 L40 基本都在高占用
不适合马上抢大任务，最好先看 nvtop/nvidia-smi
用途建议：
适合中大型训练和下游任务
适合 LCM、Brain-Semantoks、一般 fMRI/EEG 模型训练
不建议把东西放 /home，因为 /home 已经满了
个人实验建议环境优先放 /data/ningzh/envs，其余repo代码以及用到的数据都放在mnt下面的各个dataset，具体每个任务路径不同，我到时候会告诉你

4. 4090 / nccserv1 服务器   ⭐ 【新建python环境优先建在：/home/ningzh/envs ，如果空间不够我会来查询有哪些可以清掉；home/只放环境】
连接方式：ssh 4090
"C:\Users\Lenovo\.ssh\id_ed25519_4090.pub"
资源情况：
Host：nccserv1
系统：Ubuntu 22.04.4 LTS
GPU：5 张 NVIDIA GeForce RTX 4090
显存：每张约 24 GB
CPU：AMD EPYC 7T83 64-Core Processor
CPU 核心：128 CPUs
内存：503 GB
主要存储：
/：1.8 TB，已用 10%
/home：6.9 TB，已用 90%
/mnt/dataset0：143 TB，已用 91%
/mnt/dataset1：100 TB，已满，100%
/mnt/dataset2：53 TB，已用 86%
/mnt/dataset3：158 TB，已用 78%
/mnt/dataset4：415 TB，已用 97%
环境情况：
conda 可用
常用环境包括：
/home/ningzh/envs/bsem
/home/ningzh/envs/lcm
/home/ningzh/envs/mindeye2_4090_py311
当前 base 环境没有 torch
nvcc 版本：CUDA 12.2
nvidia-smi 显示驱动支持 CUDA 12.5

用途建议：
适合中等规模训练、下游分类、特征提取、debug
适合 EEG 模型、LCM/Brain-Semantoks 的小规模实验
对于显存需求很大的任务，不如 A800 稳
/home 已经 90%，不要放太大的数据或结果
个人实验建议环境优先放 /data/ningzh/envs，其余repo代码以及用到的数据都放在mnt下面的各个dataset，具体每个任务路径不同，我到时候会告诉你

