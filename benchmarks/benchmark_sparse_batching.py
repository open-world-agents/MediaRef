"""Compare native and manually chunked sparse decoder requests."""

from __future__ import annotations

import argparse
import json
import statistics
import time

from mediaref.batch import _get_decoder_class, _split_by_gap


def _run(path: str, backend: str, timestamps: list[float], strategy: str) -> None:
    decoder_class = _get_decoder_class(backend)
    with decoder_class(path) as decoder:
        if strategy == "native":
            decoder.get_frames_played_at(timestamps)
            return
        chunks = _split_by_gap(list(range(len(timestamps))), timestamps, gap_threshold=2.0)
        for _, chunk_timestamps in chunks:
            decoder.get_frames_played_at(chunk_timestamps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--backend", choices=("pyav", "torchcodec"), default="torchcodec")
    parser.add_argument("--timestamps", type=float, nargs="+", default=[0.0, 10.0, 30.0, 60.0])
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    results = {}
    for strategy in ("native", "chunked"):
        samples = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            _run(args.path, args.backend, args.timestamps, strategy)
            samples.append(time.perf_counter() - started)
        results[strategy] = {
            "median_seconds": statistics.median(samples),
            "samples_seconds": samples,
        }
    print(json.dumps({"backend": args.backend, "timestamps": args.timestamps, "results": results}, indent=2))


if __name__ == "__main__":
    main()
