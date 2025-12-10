import argparse
from agent.trainer import train


def str2bool(value):
    if isinstance(value, bool):
        return value
    lower = value.lower()
    if lower in {"yes", "true", "t", "1"}:
        return True
    if lower in {"no", "false", "f", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--replay_size", type=int, default=50000, help="Reply buffer size")
    parser.add_argument("--replay_start_size", type=int, default=5000, help="Replay start size")
    parser.add_argument("--sync_target_frames", type=int, default=2000, help="Sync target network frames")
    parser.add_argument("--epsilon_decay_last_frame", type=int, default=150000, help="Epsilon decay last frame")
    parser.add_argument("--total_steps", type=int, default=300000, help="Total training steps")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Epsilon start value")
    parser.add_argument("--epsilon_final", type=float, default=0.05, help="Epsilon final value")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for training")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use if CUDA is enabled")
    parser.add_argument(
        "--use_prior",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable prioritized experience replay.",
    )
    parser.add_argument(
        "--use_double",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable Double DQN target computation.",
    )
    parser.add_argument(
        "--use_dueling",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable the dueling network architecture.",
    )
    parser.add_argument(
        "--use_noisy",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable Noisy Net for exploration.",
    )
    parser.add_argument(
        "--sigma_init",
        type=float,
        default=0.5,
        help="Initial sigma for NoisyNet (sigma_init).",
    )
    parser.add_argument(
        "--use_simulation",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable SimPy simulation for reward (slower but more accurate).",
    )
    parser.add_argument(
        "--simulation_duration",
        type=float,
        default=2000,
        help="Simulation duration in time units (default: 86400 for extended production).",
    )
    parser.add_argument(
        "--reward_decompose",
        type=str,
        choices=["none", "mean", "discount"],
        default="none",
        help="奖励分解方式: none(默认,原逻辑), mean(平均分配), discount(折扣分配)"
    )
    parser.add_argument(
        "--reward_gamma",
        type=float,
        default=0.9,
        help="奖励分解时的折扣因子，仅discount模式有效"
    )
    # 奖励权重参数 权重会自动归一化，不需要手动归一化
    parser.add_argument(
        "--weight_distance",
        type=float,
        default=0.20,
        help="运输距离权重 (transportation_intensity)"
    )
    parser.add_argument(
        "--weight_logistics",
        type=float,
        default=0.30,
        help="物流强度权重 (material_flow_clarity)"
    )
    parser.add_argument(
        "--weight_flow",
        type=float,
        default=0.20,
        help="物料流清晰度权重 (flow_clarity/space_utilization)"
    )
    parser.add_argument(
        "--weight_throughput",
        type=float,
        default=0.25,
        help="吞吐量权重 (throughput_time)"
    )
    parser.add_argument(
        "--weight_utilization",
        type=float,
        default=0.05,
        help="设备利用率权重 (utilization)"
    )
    # 摆放顺序参数
    parser.add_argument(
        "--placement_order",
        type=str,
        choices=["default" , "process_flow", "logistics_intensity", "size_desc", "size_asc", "flow_desc", "random"],
        default="default",
        help="摆放顺序: default(配置文件顺序), process_flow(按工艺流程), logistics_intensity(按物流强度), size_desc(面积从大到小), size_asc(面积从小到大), flow_desc(物料连接数从多到少), random(随机)"
    )
    # 校准参数
    parser.add_argument(
        "--calibrate_episodes",
        type=int,
        default=0,
        help="校准回合数：运行N次随机摆放来估计指标边界。0表示不校准，使用默认硬编码边界。建议值：100"
    )
    parser.add_argument(
        "--throughput_target",
        type=float,
        default=None,
        help="用户指定的吞吐量目标（产品数）。如果不指定，则通过校准自动估计"
    )
    args = parser.parse_args()

    train(args)
