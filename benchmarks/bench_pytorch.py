#!/usr/bin/env python3
"""Optional eager-CPU reference for TensorLib benchmark shapes."""

import argparse
import csv
import math
import statistics
import time

try:
    import torch
except ImportError as error:
    raise SystemExit("PyTorch is not installed; skipping optional baseline") from error


PROFILES = {
    "smoke": (1, 0.0, 0.0),
    "quick": (5, 0.02, 0.02),
    "full": (15, 0.1, 0.25),
}


def measure(operation, profile):
    samples, minimum, warmup = PROFILES[profile]
    elapsed = 0.0
    checksum = 0.0
    while elapsed < warmup:
        start = time.perf_counter()
        checksum += operation()
        elapsed += time.perf_counter() - start
    start = time.perf_counter()
    checksum += operation()
    single = time.perf_counter() - start
    iterations = min(1_000_000, max(1, math.ceil(minimum / single))) if single else 1
    timings = []
    checksum = 0.0
    for _ in range(samples):
        start = time.perf_counter()
        for _ in range(iterations):
            checksum += operation()
        timings.append((time.perf_counter() - start) / iterations)
    timings.sort()
    return timings[len(timings) // 2], timings[math.ceil(0.95 * len(timings)) - 1], checksum, iterations


class TinyDecoder(torch.nn.Module):
    def __init__(self, time_steps, channels, layers, dropout):
        super().__init__()
        self.token = torch.nn.Embedding(256, channels)
        self.position = torch.nn.Embedding(time_steps, channels)
        self.blocks = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                channels, 6, channels * 4, dropout, "gelu",
                batch_first=True, norm_first=True)
            for _ in range(layers)
        ])
        self.norm = torch.nn.LayerNorm(channels)
        self.head = torch.nn.Linear(channels, 256)

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1])
        value = self.token(tokens) + self.position(positions)
        mask = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1],
                                     dtype=torch.bool), diagonal=1)
        for block in self.blocks:
            value = block(value, src_mask=mask)
        return self.head(self.norm(value))


def forward_operation(model, inputs):
    def operation():
        output = model(inputs)
        result = float(output.reshape(-1)[0])
        del output
        return result
    return operation


def train_operation(model, inputs, targets, optimizer):
    def operation():
        output = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            output.reshape(-1, output.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return float(loss.detach())
    return operation


def tensor_operation(function):
    def operation():
        output = function()
        result = float(output.reshape(-1)[0])
        del output
        return result
    return operation


def run_case(writer, args, name, shape, metric, units, operation):
    median, p95, checksum, iterations = measure(operation, args.profile)
    value = units / median
    print(f"  {name:24} threads={args.threads:<3} median={median * 1000:9.3f} ms "
          f"{metric}={value:9.3f}")
    if writer:
        writer.writerow(["pytorch", name, shape, "eager-float32", args.profile,
                         args.threads, torch.get_num_threads(), median, p95,
                         metric, value, iterations, checksum, "ok"])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=PROFILES, default="quick")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--csv")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.profile = "smoke"
    if args.threads < 1:
        raise SystemExit("--threads must be positive")
    torch.manual_seed(0xB34C4)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    smoke = args.profile == "smoke"
    full = args.profile == "full"
    square = 32 if smoke else (1024 if full else 256)
    tokens = 8 if smoke else (512 if full else 128)
    batch = 1 if smoke else (4 if full else 2)
    time_steps = 8 if smoke else 128
    channels = 24 if smoke else 192
    layers = 1 if smoke else 4
    output = open(args.csv, "w", newline="", encoding="utf-8") if args.csv else None
    writer = csv.writer(output) if output else None
    if writer:
        writer.writerow(["suite", "case", "shape", "layout", "profile",
                         "requested_threads", "actual_threads", "median_seconds",
                         "p95_seconds", "metric", "value", "iterations",
                         "checksum", "status"])

    print("PyTorch eager CPU baseline")
    a = torch.randn(square, square)
    b = torch.randn(square, square)
    run_case(writer, args, "matmul_square", "[MxK]x[KxN]", "GFLOP/s",
             2.0 * square ** 3 / 1e9, tensor_operation(lambda: a @ b))
    qa = torch.randn(tokens, 192)
    qb = torch.randn(192, 576)
    run_case(writer, args, "matmul_qkv", "[BTxC]x[Cx3C]", "GFLOP/s",
             2.0 * tokens * 192 * 576 / 1e9,
             tensor_operation(lambda: qa @ qb))
    attention_batch = 2 if smoke else 12
    attention_time = 16 if smoke else 128
    attention_width = 8 if smoke else 64
    aa = torch.randn(attention_batch, attention_time, attention_width)
    ab = torch.randn(attention_batch, attention_width, attention_time)
    run_case(writer, args, "matmul_attention", "[BHxTxD]x[BHxDxT]", "GFLOP/s",
             2.0 * attention_batch * attention_time ** 2 * attention_width / 1e9,
             tensor_operation(lambda: aa @ ab))

    mlp_input = torch.randn(2 if smoke else 64, 784)
    mlp_targets = torch.arange(mlp_input.shape[0]) % 10
    mlp = torch.nn.Sequential(torch.nn.Linear(784, 128), torch.nn.ReLU(),
                              torch.nn.Linear(128, 10))
    run_case(writer, args, "mnist_mlp_forward", "[Bx784]->[Bx10]", "samples/s",
             float(mlp_input.shape[0]), forward_operation(mlp, mlp_input))
    mlp_train = torch.nn.Sequential(torch.nn.Linear(784, 128), torch.nn.ReLU(),
                                    torch.nn.Linear(128, 10))
    run_case(writer, args, "mnist_mlp_train_step", "[Bx784]->[Bx10]", "samples/s",
             float(mlp_input.shape[0]), train_operation(
                 mlp_train, mlp_input, mlp_targets,
                 torch.optim.SGD(mlp_train.parameters(), lr=0.05)))

    decoder_input = torch.arange(batch * time_steps).reshape(batch, time_steps) % 256
    decoder_targets = decoder_input.clone()
    decoder = TinyDecoder(time_steps, channels, layers, 0.0).eval()
    run_case(writer, args, "tiny_lm_forward", "[BxT]->[BxTx256]", "tokens/s",
             float(batch * time_steps), forward_operation(decoder, decoder_input))
    decoder_train = TinyDecoder(time_steps, channels, layers, 0.1).train()
    run_case(writer, args, "tiny_lm_train_step", "[BxT]->[BxTx256]", "tokens/s",
             float(batch * time_steps), train_operation(
                 decoder_train, decoder_input, decoder_targets,
                 torch.optim.AdamW(decoder_train.parameters(), lr=1e-3)))
    if output:
        output.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
