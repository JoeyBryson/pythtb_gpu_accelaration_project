# Scientific Parallelism Hackathon

SoTA (Society for Technological Advancement), 21st–22nd November 2026, King's Cross, London.
Supported by Callosum and Inherent Labs, along with ARIA and Fractile.

## Logistics

- Teams can form on day one or arrive as a team. **All members must attend in person.**
- Provided: GPU access, AI/API credits, and support from judges and mentors across computational science,
  performance engineering, applied maths, and AI for science.
- Prize pool £8,500:
  - **1st:** £5,000 + a demo slot at ARIA's 2026 AI for Science Summit (26th November, Manchester)
  - **2nd:** £2,500
  - **3rd:** £1,000

## Challenge

- Choose a scientifically relevant computational problem from research, an existing codebase, or another field.
- Redesign its solution for massively parallel hardware. Consider how work is divided, how data moves, and which
  operations can run concurrently.
- Build the solution and benchmark it against a meaningful baseline. Measure and explain its performance.
- Show what scientific investigation the approach enables, such as more experiments, higher resolution, improved
  uncertainty estimates, or a new capability.

The organisers stress going beyond raw speed-ups: **rethink and redesign the underlying logical chain**, i.e. how
data flows, which operations can run at the same time, which bottlenecks go away, and which different algorithms
become possible when millions of operations run in parallel. The goal is to exploit the architecture itself (a new
algorithm, numerical method or computational structure) and to measure and explain the impact on a scientific
challenge.

Their framing: most scientific software assumes serial CPU execution. GPUs trade a few capable cores for thousands
of simpler ones, a different cost structure that favours a different family of algorithms.

Example challenges they gave (we're bringing our own): flood uncertainty (thousands of rainfall scenarios),
faint asteroid detection (stacking images along motion paths), molecular pose search (millions of poses in a
protein pocket).

## Possible routes (from the brief)

- **Expose existing parallelism:** many independent computations (Monte Carlo, parameter sweeps, particle
  methods, image pipelines). *Ours: thousands of disorder configurations × k-points × k-lines.*
- **Rethink existing algorithms:** the fastest parallel method may do more total work than the fastest serial
  one. *Ours: Newton-Schulz instead of SVD, tree reduction instead of a sequential product.*
- **Change the representation:** data layouts or mathematical representations that map better onto a GPU.
  *Ours: stacking everything into batched tensors with leading batch dimensions.*
- **Cross a computational threshold:** acceleration that changes what can be done (hours → interactive,
  offline → real time, newly feasible scale). *Ours: a full disorder-averaged phase diagram with error bars
  becomes a minutes-scale run.*

## Judging criteria

Strong submissions demonstrate some combination of:

- **Correctness and assumptions:** Preserve intended behaviour, or clearly state and justify assumptions
  introduced to make the problem tractable.
- **Fundamental reshaping:** Creatively restructure the problem, algorithm, or data flow to expose parallelism.
- **Meaningful performance improvement:** Demonstrate a clear speedup against a credible, relevant baseline and
  explain the measurements.
- **Scientific value:** Useful results, sound methods, and convincing technical evidence. **Informative negative
  results are also valuable**: a carefully explained failure to achieve a speed-up can still win a prize
  (e.g. our finding that GPU SVD is slower than CPU is worth explaining, not hiding).
