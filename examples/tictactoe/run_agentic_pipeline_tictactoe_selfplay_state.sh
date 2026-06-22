#!/bin/bash
set +x
ray stop

CONFIG_PATH=$(basename $(dirname $0))

ROLL_PATH=${PWD}
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

# transformer-engine 必须设这些环境变量（megatron 训练侧依赖 TE）
export CUDA_HOME=/usr/local/cuda
export NVTE_FRAMEWORK=pytorch
export PATH=$CUDA_HOME/bin:$PATH

# 显存碎片优化（缓解 OOM）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROLL_OUTPUT_DIR="./runs/tictactoe_selfplay_state/$(date +%Y%m%d-%H%M%S)"
ROLL_LOG_DIR=$ROLL_OUTPUT_DIR/logs
ROLL_RENDER_DIR=$ROLL_OUTPUT_DIR/render
export ROLL_OUTPUT_DIR=$ROLL_OUTPUT_DIR
export ROLL_LOG_DIR=$ROLL_LOG_DIR
export ROLL_RENDER_DIR=$ROLL_RENDER_DIR
mkdir -p $ROLL_LOG_DIR $ROLL_RENDER_DIR

python examples/start_agentic_pipeline.py --config_path $CONFIG_PATH  --config_name agentic_val_tictactoe_selfplay_state | tee $ROLL_LOG_DIR/custom_logs.log
