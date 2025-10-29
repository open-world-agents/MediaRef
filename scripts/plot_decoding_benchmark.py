#!/usr/bin/env python3
"""Generate benchmark comparison figure for README."""

import matplotlib.pyplot as plt
import numpy as np

# Benchmark data
configurations = ["Baseline", "TorchCodec\n(batch)", "MediaRef\n(adaptive batch)"]
throughput = [24.25, 79.73, 119.16]  # img/s
io_efficiency = [41.69, 770.39, 18.73]  # KB/img

# Convert KB/img to img/KB for more intuitive visualization
io_efficiency_inverted = [1000 / x for x in io_efficiency]  # img/KB

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Colors
colors = ["#6B7280", "#F59E0B", "#10B981"]

# Plot 1: Throughput
bars1 = ax1.bar(configurations, throughput, color=colors, alpha=0.8, edgecolor="black", linewidth=1.2)
ax1.set_ylabel("Throughput (img/s)", fontsize=12, fontweight="bold")
ax1.set_title("Decoding Throughput", fontsize=13, fontweight="bold", pad=15)
ax1.grid(axis="y", alpha=0.3, linestyle="--")
ax1.set_axisbelow(True)

# Add value labels on bars
for bar, val in zip(bars1, throughput):
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

# Add speedup annotations
ax1.text(
    1, throughput[1] * 0.5, "3.3×", ha="center", va="center", fontsize=10, color="white", fontweight="bold"
)
ax1.text(
    2, throughput[2] * 0.5, "4.9×", ha="center", va="center", fontsize=10, color="white", fontweight="bold"
)

# Plot 2: I/O Efficiency (inverted to img/KB)
bars2 = ax2.bar(configurations, io_efficiency_inverted, color=colors, alpha=0.8, edgecolor="black", linewidth=1.2)
ax2.set_ylabel("I/O Efficiency (img/KB)", fontsize=12, fontweight="bold")
ax2.set_title("I/O Efficiency", fontsize=13, fontweight="bold", pad=15)
ax2.grid(axis="y", alpha=0.3, linestyle="--")
ax2.set_axisbelow(True)

# Add value labels on bars
for bar, val in zip(bars2, io_efficiency_inverted):
    height = bar.get_height()
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{val:.1f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

# Add efficiency improvement annotations
ax2.text(
    1,
    io_efficiency_inverted[1] * 0.5,
    "0.03×",
    ha="center",
    va="center",
    fontsize=10,
    color="white",
    fontweight="bold",
)
ax2.text(
    2,
    io_efficiency_inverted[2] * 0.5,
    "2.2×",
    ha="center",
    va="center",
    fontsize=10,
    color="white",
    fontweight="bold",
)

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig("assets/decoding_benchmark.png", dpi=150, bbox_inches="tight")
print("Figure saved to assets/decoding_benchmark.png")

