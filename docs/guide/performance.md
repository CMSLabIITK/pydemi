# Speed and memory

Measured on one CPU core (Linux workstation), nearest-atom partition, descriptors plus the
plot data of the interface:

| CHGCAR | grid | time | peak memory |
|---|---|---|---|
| FeNi₃, 6 MB | 56³ | 0.7 s | 0.17 GB |
| Al₂Te₅, 31 MB | 64 × 108 × 128 | 4.1 s | 0.38 GB |
| SnBi₂Te₄, 350 MB | 216³ | 54 s | 2.8 GB |

The cost grows roughly with the number of grid points. The Becke partition takes about ten
times longer on large grids.

## Many structures

`pydemi batch` runs one process per structure, so the peak memory of a run is about
`workers` × the peak of the largest structure in flight. For a dataset of mixed sizes, choose
`--workers` from your memory, not from your core count. `float32=True` (`--float32`) halves
the memory of the derivative fields at a negligible change of the values
({doc}`numerics`).

## Predicted densities

ChargE3Net's cost grows with the number of atom–grid-point pairs within 4 Å. On one CPU core
it takes about 0.35 ms per pair: a 40³ grid of a two-atom cell (BeCo, 2 million pairs) took
12 minutes. All cores, and above all a GPU, make it much faster. Memory is about 160 kB per
pair in flight; pydemi works through the grid in chunks sized to stay near 2.6 GB on a CPU,
and halves the chunk if memory runs out. `--grid n1,n2,n3` gives a quick coarse preview.
