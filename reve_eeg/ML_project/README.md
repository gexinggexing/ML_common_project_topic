这次我们是要完成《机器学习与医学工程应用》这门课的期末的course project（公选题部分），老师的要求就是：老师给我们提供了五个eeg的数据集  BCIC2A CHINESE MDD SEED SLEEP  ，我们需要自己去选择eeg的模型，然后在这几个数据集上刷分，就是刷最后的指标，指标越好分数越高（不需要重新预训练，用人家已经公开的预训练权重，在上面分别根据这五个数据集分别进行 ft/mlp/lp 就行，注意是分别，不是所有数据集一起训）。

可用算力资源：见 \\10.16.93.90\dataset3\nzh\eeg_FM\reve_eeg\Lenevo可用算力资源汇总.md

老师提供的需要刷分的数据集路径：（在实验室nas上，hs无法直接访问，如有需要再说，我把数据移动上去；l40和4090可以访问，这两个服务器挂载把nas挂载在mnt/下面；我本地电脑把dataset3映射成了X盘）
X:\panxy\course\project1_data\course project\course project
\\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project
/mnt/dataset3/panxy/course/project1_data/course project/course project

（本地5080版）：
dataset_info_fixed.json    \\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project\BCIC2A
test_x_only.h5             \\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project\BCIC2A
train.h5                   \\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project\BCIC2A
val.h5                     \\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project\BCIC2A

dataset_info.json          \\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project\CHINESE
test_x_only.h5             \\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project\CHINESE
train.h5                   \\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project\CHINESE
val.h5                     \\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project\CHINESE

dataset_info.json          \\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project\MDD
test_x_only.h5             \\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project\MDD
train.h5                   \\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project\MDD
val.h5                     \\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project\MDD

dataset_info.json          \\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project\SEED
dataset_info.json          \\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project\SEED\SEED
sub_1.h5                   \\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project\SEED\SEED
test_x_only.h5             \\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project\SEED
train.h5                   \\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project\SEED
val.h5                     \\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project\SEED

dataset_info.json          \\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project\SLEEP
test_x_only.h5             \\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project\SLEEP
train.h5                   \\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project\SLEEP
val.h5                     \\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project\SLEEP

TEST_DATASET.py            \\10.16.93.90\dataset3\panxy\course\project1_data\course project\course project

4090/l40版：
换成mnt/即可

hs火山版：
/vePFS-0x0d/nzh/data/eeg/ML_project  这个层级下面之后是每个数据集对应的文件夹，跟nas上分布一样



split和label（已经包含在上面的数据集组织形式里了）：以SEED为例
"X:\panxy\course\project1_data\course project\course project\SEED\test_x_only.h5"
"X:\panxy\course\project1_data\course project\course project\SEED\train.h5"
"X:\panxy\course\project1_data\course project\course project\SEED\val.h5"
由此可知split；
每个数据集的train和val的label都已经在.h5文件里了，你自己扫描一下，你最好把每个数据集的数据文件内容每个数据集整理出一个文档来（简单整理即可，放到data/下面），把他们的mnt路径也统计一下放到里面。


reve-eeg论文：见 \\10.16.93.90\dataset3\nzh\eeg_FM\reve_eeg\reve_eeg.pdf

GItHub上repo网址：
论文原始repo：https://github.com/elouayas/reve_eeg
我fork的repo： https://github.com/hvr710/reve_eeg  （不在main分支改动，我每次都会新建branch，这次我新建的seed1，我把这个branch clone到了\\10.16.93.90\dataset3\nzh\eeg_FM\reve_eeg）

本地repo路径：
\\10.16.93.90\dataset3\nzh\eeg_FM\reve_eeg

HF上的预训练权重我从hf上拉下来了，在：
\\10.16.93.90\dataset3\nzh\eeg_FM\reve_eeg\checkpoints\reve-base
\\10.16.93.90\dataset3\nzh\eeg_FM\reve_eeg\checkpoints\reve-positions


