#FLY_FORWARD_DISABLE_ACC_LOAD=1 /opt/IsaacLab/isaaclab.sh -p scripts/skrl/train.py --task=Isaac-fly-forward-v0 --headless --num_envs=1280 --algorithm="PPO"

#FLY_FORWARD_DISABLE_ACC_LOAD=1 /opt/IsaacLab/isaaclab.sh -p scripts/skrl/train.py --task=Isaac-fly-forward-v0  --num_envs=1 --algorithm="PPO"

FLY_FORWARD_DISABLE_ACC_LOAD=1 /opt/IsaacLab/isaaclab.sh -p scripts/skrl/train.py \
  --task=Isaac-fly-forward-v0 \
  --headless \
  --livestream 0 \
  --num_envs=256 \
  --algorithm=PPO

# 如果要用 4 张 4090 多卡分布式训练，用：

#  FLY_FORWARD_DISABLE_ACC_LOAD=1 /opt/IsaacLab/isaaclab.sh -p -m torch.distributed.run \
#    --standalone \
#    --nnodes=1 \
#    --nproc_per_node=4 \
#    scripts/skrl/train.py \
#    --task=Isaac-fly-forward-v0 \
#    --headless \
#    --livestream 0 \
#    --num_envs=64 \
#    --algorithm=PPO \
#    --distributed