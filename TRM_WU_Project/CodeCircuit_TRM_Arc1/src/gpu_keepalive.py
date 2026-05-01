#!/usr/bin/env python
import argparse
import signal
import sys
import time

import torch


running = True


def handle_signal(signum, frame):
    global running
    running = False


def main():
    parser = argparse.ArgumentParser(
        description="Transparent CUDA keepalive for CPU-heavy jobs that must hold GPUs."
    )
    parser.add_argument("--matrix-size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument("--log-interval", type=float, default=60.0)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        print("CUDA is not available; keepalive exiting.", flush=True)
        return 0

    device = torch.device("cuda:0")
    print(
        "Starting GPU keepalive "
        f"device={torch.cuda.get_device_name(device)} "
        f"matrix_size={args.matrix_size} steps={args.steps} sleep={args.sleep}",
        flush=True,
    )

    matrix_size = args.matrix_size
    while True:
        try:
            a = torch.randn((matrix_size, matrix_size), device=device)
            b = torch.randn((matrix_size, matrix_size), device=device)
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or matrix_size <= 1024:
                raise
            torch.cuda.empty_cache()
            matrix_size //= 2
            print(
                f"Keepalive allocation hit OOM; retrying with matrix_size={matrix_size}",
                flush=True,
            )

    if matrix_size != args.matrix_size:
        print(f"Using fallback matrix_size={matrix_size}", flush=True)

    last_log = time.time()
    loops = 0

    while running:
        for _ in range(args.steps):
            c = torch.mm(a, b)
            torch.cuda.synchronize(device)
        loops += 1

        now = time.time()
        if now - last_log >= args.log_interval:
            checksum = float(c[0, 0].detach().cpu())
            print(
                f"keepalive loops={loops} checksum={checksum:.6f} "
                f"timestamp={time.strftime('%Y-%m-%d %H:%M:%S')}",
                flush=True,
            )
            last_log = now

        time.sleep(args.sleep)

    print("GPU keepalive stopping.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
