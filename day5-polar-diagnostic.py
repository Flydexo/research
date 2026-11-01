"""Small CPU diagnostic for exact partial polar and Muon's five-step NS map."""

import json
import platform
import statistics
import time
from pathlib import Path

import torch
import trackio


A, B, C = 3.4445, -4.7750, 2.0315
STEPS = 5
REPEATS = 200
torch.set_num_threads(1)


def exact_partial_polar(g):
    u, singular_values, vh = torch.linalg.svd(g.to(torch.float64), full_matrices=False)
    rank = int((singular_values > 1e-12).sum().item())
    return u[:, :rank] @ vh[:rank, :], rank


def newton_schulz_five(g, dtype):
    x = g.to(dtype)
    x = x / (x.norm() + 1e-7)
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    for _ in range(STEPS):
        gram = x @ x.T
        x = A * x + (B * gram + C * (gram @ gram)) @ x
    return x.T if transpose else x


def median_us(fn):
    fn()  # warm-up
    timings = []
    for _ in range(REPEATS):
        start = time.perf_counter_ns()
        fn()
        timings.append((time.perf_counter_ns() - start) / 1_000)
    return statistics.median(timings)


def analyze(name, g, dtype):
    exact, rank = exact_partial_polar(g)
    approx_raw = newton_schulz_five(g, dtype)
    approx = approx_raw.to(torch.float64)
    g64 = g.to(torch.float64)
    nuclear = float(torch.linalg.svdvals(g64).sum())
    alignment = float((g64 * approx).sum())
    op_norm = float(torch.linalg.svdvals(approx).max())
    result = {
        "case": name,
        "shape": list(g.shape),
        "rank": rank,
        "input_dtype": str(g.dtype),
        "ns_dtype": str(approx_raw.dtype),
        "svd_reference_dtype": str(exact.dtype),
        "input_singular_values": torch.linalg.svdvals(g64).tolist(),
        "exact_singular_values": torch.linalg.svdvals(exact).tolist(),
        "ns_singular_values": torch.linalg.svdvals(approx).tolist(),
        "descent_pairing_exact": float((g64 * exact).sum()),
        "descent_pairing_ns": alignment,
        "spectral_ball_alignment_ratio": alignment / (nuclear * op_norm),
        "frobenius_error_to_partial_polar": float(torch.linalg.norm(approx - exact)),
        "exact_svd_median_us": median_us(lambda: exact_partial_polar(g)),
        "ns_median_us": median_us(lambda: newton_schulz_five(g, dtype)),
    }
    trackio.init(
        project="day5_polar_diagnostic",
        name=f"{name}_{str(dtype).split('.')[-1]}",
        config={
            "case": name,
            "shape": str(tuple(g.shape)),
            "rank": rank,
            "dtype": str(dtype),
            "ns_steps": STEPS,
            "coefficients": str((A, B, C)),
            "repeats": REPEATS,
            "device": "cpu",
            "reference": "reduced-SVD partial polar, float64",
        },
        auto_log_cpu=False,
        auto_log_gpu=False,
    )
    trackio.log(
        {
            "descent_pairing_exact": result["descent_pairing_exact"],
            "descent_pairing_ns": alignment,
            "spectral_ball_alignment_ratio": result["spectral_ball_alignment_ratio"],
            "frobenius_error_to_partial_polar": result["frobenius_error_to_partial_polar"],
            "ns_operator_norm": op_norm,
            "exact_svd_median_us": result["exact_svd_median_us"],
            "ns_median_us": result["ns_median_us"],
            **{f"input_sigma_{i}": v for i, v in enumerate(result["input_singular_values"])},
            **{f"exact_sigma_{i}": v for i, v in enumerate(result["exact_singular_values"])},
            **{f"ns_sigma_{i}": v for i, v in enumerate(result["ns_singular_values"])},
        }
    )
    trackio.finish()
    return result


def main():
    cases = {
        "tall_full_rank": torch.tensor([[3.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),
        "wide_full_rank": torch.tensor([[3.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "tall_rank_one": torch.tensor([[3.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
    }
    results = {
        "purpose": "Educational matrix diagnostic; no model training or paid compute",
        "pytorch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": "cpu",
        "runtime_note": "Tiny CPU matrix timings are implementation overhead measurements, not GPU training benchmarks.",
        "cases": [
            analyze(name, g, dtype)
            for name, g in cases.items()
            for dtype in (torch.float32, torch.bfloat16)
        ],
    }
    output = Path(__file__).with_name("day5-polar-diagnostic-results.json")
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
