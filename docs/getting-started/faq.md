# Questions and troubles

## About the descriptors

**Why only the CHGCAR?**
It is the one file every VASP run writes, and it is enough: the structure is in its header and
the density in its first block. Needing AECCARs, an ELFCAR or a LOCPOT would rule out most
existing calculations.

**Is the density all-electron?**
No. A CHGCAR holds the PAW **valence pseudo-density**: no deep cores, smooth (and possibly
slightly negative) inside the augmentation spheres. The descriptors describe the valence
electrons as the POTCARs define them; see {doc}`../descriptors/theory`.

**Can I mix densities from different POTCARs?**
Only with care. The POTCAR sets the number of valence electrons (for example `Ti` vs `Ti_sv`),
which moves several descriptors. Keep one POTCAR set per dataset.

**Why is there no ELF, kinetic-energy density or bond-critical-point ellipticity?**
They need orbitals or are numerically unstable on VASP grids; see {ref}`not-included`.

**Why did a descriptor come out as exactly 0, 1/2 or 1/3?**
Either symmetry (cubic crystals have `T_eigenvalues` of 1/3 and `perc_anisotropy` 0) or a
special case. Check `meta["<name>__flag"]`; see {doc}`../guide/metadata`.

**Are the values the same for the primitive and the conventional cell?**
Yes, up to the sampling of the two grids: every descriptor is intensive (see
{doc}`../guide/validation`). Only the
labels of `rho_perc_a/b/c` follow the cell vectors as written.

**What about spin-polarised runs?**
The total density (the first block) is used; the magnetisation is ignored.

## Installation and running

**`pydemi gui` says a package is missing.**
Install the extra: `pip install "pydemi[gui]"` (and `,ml` for predicted densities).

**The Properties view is missing on macOS.**
LightGBM needs OpenMP: `brew install libomp`.

**PyTorch says the NVIDIA driver is too old.**
Your PyTorch was built for a newer CUDA than the driver supports. Install a matching CUDA
build of PyTorch, or run `pydemi predict … --device cpu`.

**ChargE3Net refuses a structure.**
The model knows the 87 elements of its training POTCARs only; `pydemi.ml.unsupported(s)`
lists the ones it does not know.

**A batch run is killed.**
Too many workers for the memory: each holds one density and its derived fields. Lower
`--workers`, or add `--float32` ({doc}`../guide/performance`).

**The interface runs on a remote machine.**
`pydemi gui --no-browser` there, then `ssh -L 8501:localhost:8501 host` and open
`http://localhost:8501` locally.

## Something else

Open an issue at [github.com/CMSLabIITK/pydemi/issues](https://github.com/CMSLabIITK/pydemi/issues).
