"""Extract the exact plotted data from the papers' vector figures (no digitising by eye).

Both figures are vector graphics inside the PDFs. We convert the page to SVG with poppler's
`pdftocairo`, pick the data paths by stroke colour, and map their coordinates to axis
units using the tick marks. Calibration was found by inspecting the SVG (21 Nov 2026).

  Orth et al. 2016, Fig. 5 (arXiv:1512.03233 v2, page 5): Wc vs lambda_nu, tight-binding
      points + error bars, and the lowest-order Born curve. W in ORTH's convention.
  Favata & Marrazzo 2023, Fig. 7 left (arXiv:2301.02612, page 8): mean single-point spin
      Chern number vs W, L = 15 and L = 42, +- standard deviation over configurations.
      W in OUR convention (eps in [-W/2, W/2]).

Writes analysis/reference_data/*.csv. Needs `pdftocairo` (poppler-utils).
Run from hackathon/:  uv run python analysis/extract_paper_data.py
"""

from __future__ import annotations

import csv
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPERS = HERE.parent / "joeys_og_folder" / "papers"
OUT = HERE / "reference_data"


def page_svg(pdf: Path, page: int) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        svg = Path(tmp) / "page.svg"
        subprocess.run(["pdftocairo", "-svg", "-f", str(page), "-l", str(page), str(pdf), str(svg)],
                       check=True)
        return svg.read_text()


def paths(svg: str, transform: str, stroke_prefix: str) -> list[list[float]]:
    """Coordinate lists of every path with this transform and stroke colour."""
    out = []
    for m in re.finditer(r"<path[^>]*>", svg):
        p = m.group(0)
        st = re.search(r'stroke="([^"]*)"', p)
        if f'transform="{transform}"' in p and st and st.group(1).startswith(stroke_prefix):
            d = re.search(r' d="([^"]*)"', p).group(1)
            out.append([float(v) for v in re.findall(r"-?\d+\.?\d*", d)])
    return out


def axis(p0: float, v0: float, p1: float, v1: float):
    return lambda p: v0 + (p - p0) / (p1 - p0) * (v1 - v0)


def is_vertical(n: list[float]) -> bool:
    return len(n) == 4 and abs(n[0] - n[2]) < 1e-6


def orth_fig5() -> list[dict]:
    svg = page_svg(PAPERS / "1512.03233v2.pdf", 5)
    T = "matrix(0.101237, 0, 0, -0.101237, 56.068, 465.298)"
    X = axis(241.026008, 1.4, 2292.987001, 2.1)  # frame edges = axis limits 1.4 .. 2.1
    Y = axis(284.962822, 0.0, 1545.618512, 3.0)  # ticks at 0 and 3
    rows = []
    for n in paths(svg, T, "rgb(80.46875%"):  # red: tight-binding error bars
        if is_vertical(n) and n[1] < 1290:  # legend symbol bars start at 1292 (top left)
            lo, hi = sorted((Y(n[1]), Y(n[3])))
            rows.append({"lambda_nu": round(X(n[0]), 4), "Wc_orth": round((lo + hi) / 2, 4),
                         "err_orth": round((hi - lo) / 2, 4), "source": "tight-binding"})
    born = max(paths(svg, T, "rgb(0%, 0%, 33"), key=len)  # dark blue: Born curve
    for i in range(0, len(born), 2):
        rows.append({"lambda_nu": round(X(born[i]), 4), "Wc_orth": round(Y(born[i + 1]), 4),
                     "err_orth": "", "source": "born"})
    return sorted(rows, key=lambda r: (r["source"], r["lambda_nu"]))


def favata_fig7_left() -> list[dict]:
    svg = page_svg(PAPERS / "related_work" / "pdf" / "2301.02612.pdf", 8)
    T = "matrix(0.27351, 0, 0, -0.27351, 56.749, 587.35)"
    X = axis(127.441296, 0.0, 871.444029, 12.0)  # ticks at W = 0 and 12
    Y = axis(179.746627, 0.0, 458.515571, 1.0)  # dashed lines at C = 0 and 1
    rows = []
    for L, colour in ((15, "rgb(58.03833%"), (42, "rgb(83.920288%")):
        ps = paths(svg, T, colour)
        line = max(ps, key=len)
        sd = {round(X(n[0]), 3): abs(Y(n[3]) - Y(n[1])) / 2 for n in ps if is_vertical(n)}
        for i in range(0, len(line), 2):
            w = round(X(line[i]), 3)
            rows.append({"L": L, "W": w, "C_s_mean": round(Y(line[i + 1]), 4),
                         "C_s_sd": round(sd.get(w, float("nan")), 4)})
    return rows


def save(name: str, rows: list[dict]) -> None:
    OUT.mkdir(exist_ok=True)
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT / name} ({len(rows)} rows)")


if __name__ == "__main__":
    save("orth2016_fig5.csv", orth_fig5())
    save("favata2023_fig7_left.csv", favata_fig7_left())
