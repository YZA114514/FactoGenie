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
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--replay_size", type=int, default=100000, help="Reply buffer size")
    parser.add_argument("--replay_start_size", type=int, default=10000, help="Replay start size")
    parser.add_argument("--sync_target_frames", type=int, default=1000, help="Sync target network frames")
    parser.add_argument("--epsilon_decay_last_frame", type=int, default=150000, help="Epsilon decay last frame")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Epsilon start value")
    parser.add_argument("--epsilon_final", type=float, default=0.01, help="Epsilon final value")
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
        "--use_simulation",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Enable SimPy simulation for reward (slower but more accurate).",
    )
    
    args = parser.parse_args()

    train(args)
