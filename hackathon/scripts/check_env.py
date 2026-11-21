"""Check a machine is ready: PythTB imports, torch sees the GPU, and a quick
float32 vs float64 eigh timing so we can decide the precision strategy on the
event's hardware (T4s are ~32x slower in float64; A100/H100 much less so).

Run:  uv run python scripts/check_env.py
"""

import platform
import time

import numpy as np
import pythtb
import torch

from gpu_engine import get_backend, timed


def main() -> None:
    print(f"python   {platform.python_version()} on {platform.platform()}")
    print(f"numpy    {np.__version__}")
    print(f"pythtb   {pythtb.__version__}  ({pythtb.__file__})")
    print(f"torch    {torch.__version__}, CUDA build: {torch.version.cuda}")
    print(f"cuda ok  {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
    try:
        import cupy as cp  # optional

        print(f"cupy     {cp.__version__}")
    except ImportError:
        print("cupy     not installed (optional: uv sync --extra cupy)")

    # Batched Hermitian eigh at the 8x8-supercell size: 961 matrices of 256x256.
    batch, n = 961, 256
    rng = np.random.default_rng(0)
    a = rng.standard_normal((batch, n, n)) + 1j * rng.standard_normal((batch, n, n))
    h = a + np.conj(np.swapaxes(a, -1, -2))

    t0 = time.perf_counter()
    np.linalg.eigh(h)
    print(f"\nnumpy eigh c128 ({batch}x{n}x{n}): {time.perf_counter() - t0:.2f} s")

    for precision in ("double", "single"):
        be = get_backend(precision)
        x = torch.as_tensor(h, dtype=be.cdtype, device=be.device)
        torch.linalg.eigh(x[:8])  # warm-up (kernel loading / allocation)
        with timed(verbose=False, backend=be) as t:
            torch.linalg.eigh(x)
        print(f"torch eigh on {be.describe()}: {t.seconds:.2f} s")
        del x
        if be.is_gpu:
            torch.cuda.empty_cache()

    if not torch.cuda.is_available():
        print("\nNo CUDA GPU here: fine for correctness tests, not for benchmarks.")


if __name__ == "__main__":
    main()
