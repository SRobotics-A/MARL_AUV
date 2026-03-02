from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg


def add_rsl_rl_args(parser: argparse.ArgumentParser):
    """向解析器添加RSL-RL相关参数
    
    该函数用于配置RSL-RL强化学习代理的命令行参数，包括实验设置、
    模型加载、日志记录等关键配置选项。

    Args:
        parser: 要添加参数的参数解析器对象
    """
    # 创建新的参数组，便于组织相关参数
    arg_group = parser.add_argument_group("rsl_rl", description="RSL-RL代理的相关参数配置")
    
    # -- 实验相关参数 --
    arg_group.add_argument(
        "--experiment_name", 
        type=str, 
        default=None, 
        help="实验文件夹名称，用于存储训练日志和模型检查点"
    )
    arg_group.add_argument(
        "--run_name", 
        type=str, 
        default=None, 
        help="运行名称后缀，将附加到日志目录名中"
    )
    
    # -- 模型加载参数 --
    arg_group.add_argument(
        "--resume", 
        type=bool, 
        default=None, 
        help="是否从检查点恢复训练"
    )
    arg_group.add_argument(
        "--load_run", 
        type=str, 
        default=None, 
        help="要从中恢复的运行文件夹名称"
    )
    arg_group.add_argument(
        "--checkpoint", 
        type=str, 
        default=None, 
        help="要恢复的具体检查点文件路径"
    )
    
    # -- 日志记录参数 --
    arg_group.add_argument(
        "--logger", 
        type=str, 
        default=None, 
        choices={"wandb", "tensorboard", "neptune"}, 
        help="要使用的日志记录模块：wandb、tensorboard或neptune"
    )
    arg_group.add_argument(
        "--log_project_name", 
        type=str, 
        default=None, 
        help="使用wandb或neptune时的日志项目名称"
    )


def parse_rsl_rl_cfg(task_name: str, args_cli: argparse.Namespace) -> RslRlOnPolicyRunnerCfg:
    """根据输入参数解析RSL-RL代理配置
    
    该函数加载默认配置并根据命令行参数进行覆盖，
    实现灵活的配置管理机制。

    Args:
        task_name: 环境任务名称
        args_cli: 命令行参数命名空间对象

    Returns:
        基于输入参数解析后的RSL-RL代理配置对象
    """
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    # 从注册表加载默认配置
    rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")

    # 使用CLI参数覆盖默认配置
    if args_cli.seed is not None:
        rslrl_cfg.seed = args_cli.seed  # 随机种子设置
    if args_cli.resume is not None:
        rslrl_cfg.resume = args_cli.resume  # 恢复训练标志
    if args_cli.load_run is not None:
        rslrl_cfg.load_run = args_cli.load_run  # 加载运行名称
    if args_cli.checkpoint is not None:
        rslrl_cfg.load_checkpoint = args_cli.checkpoint  # 检查点文件路径
    if args_cli.run_name is not None:
        rslrl_cfg.run_name = args_cli.run_name  # 运行名称
    if args_cli.logger is not None:
        rslrl_cfg.logger = args_cli.logger  # 日志记录器类型
    
    # 为wandb和neptune设置项目名称
    if rslrl_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
        rslrl_cfg.wandb_project = args_cli.log_project_name  # WandB项目名称
        rslrl_cfg.neptune_project = args_cli.log_project_name  # Neptune项目名称

    return rslrl_cfg
