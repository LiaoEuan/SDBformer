#!/bin/bash

# ===================================================================
# Slurm SBATCH 参数设置: 为您的LOSO实验量身定制
# ===================================================================
#
#SBATCH -J LOSO_DANN_AAD         # 作业名称 (Job Name)，清晰明了
#SBATCH -o %x-%j.out             # 标准输出日志, %x=作业名, %j=作业ID
#SBATCH -e %x-%j.err             # 标准错误日志
#
# --- 资源申请 ---
# 您的实验包含18次完整的训练，会耗时较长，请确保资源和时间充足
#
#SBATCH -p gpu2node              # 提交到gpu2node分区 (根据需要可改为gpu3node)
#SBATCH -c 8                     # 申请8个CPU核心 (用于数据加载 num_workers)
#SBATCH --gres=gpu:1             # 申请1块GPU
#SBATCH --mem=48G                # 申请48GB内存 (考虑到长时间运行和数据缓存，适当增加)
#SBATCH -t 2-00:00:00            # 最长运行时间: 2天 (格式: D-HH:MM:SS)

# ===================================================================
# 任务执行前的准备工作
# ===================================================================
# 打印一些有用的信息到日志文件，方便追踪
echo "========================================================"
echo "作业启动 (Job Start)"
echo "作业名 (Job Name): $SLURM_JOB_NAME"
echo "作业ID (Job ID): $SLURM_JOB_ID"
echo "运行节点 (Node List): $SLURM_JOB_NODELIST"
echo "起始时间 (Start Time): $(date)"
echo "--------------------------------------------------------"

# --- 关键步骤: 进入代码所在目录 ---
# Slurm作业默认从您提交命令时所在的目录开始执行,
# 明确切换到代码目录是一个非常好的习惯, 可以避免路径问题。
cd /share/home/yuan/DTU_KUL_LOSO
echo "当前工作目录 (Working Directory): $(pwd)"
echo "--------------------------------------------------------"


# 1. 清理并加载所需模块
echo "正在加载环境模块 (cuda, anaconda3)..."
module purge
module load cuda/11.8
module load anaconda3
# ... (module load anaconda3 的后面) ...

# --- 2. 激活您的Conda环境 (更健壮的方式) ---
# 首先，找到conda的安装位置并初始化shell
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# 然后，激活您指定的环境
CONDA_ENV_NAME="ly_torch"
echo "正在激活 Conda 环境: $CONDA_ENV_NAME..."
conda activate $CONDA_ENV_NAME
if [ $? -ne 0 ]; then
    echo "错误: 激活Conda环境 '$CONDA_ENV_NAME' 失败！请检查环境名称。"
    exit 1
fi
echo "Conda 环境激活成功。当前Python路径: $(which python)"



# ===================================================================
# 核心任务执行
# ===================================================================
echo "--------------------------------------------------------"
echo "开始执行Python LOSO脚本 (testdann_keep1.py)..."
echo "这将会是一个漫长的过程，请耐心等待。"
echo "--------------------------------------------------------"

# 执行您的Python脚本
# -u 参数确保Python的输出不经过缓冲，直接写入日志文件，方便实时查看进度
/share/home/yuan/.conda/envs/ly_torch/bin/python -u DE_model_SNN_V3.py



# ===================================================================
# 任务执行后的收尾工作
# ===================================================================
echo "--------------------------------------------------------"
echo "Python脚本执行完毕，退出码 (Exit Code): $?"
echo "结束时间 (End Time): $(date)"
echo "作业结束 (Job End)"
echo "========================================================"