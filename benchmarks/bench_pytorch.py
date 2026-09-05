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
