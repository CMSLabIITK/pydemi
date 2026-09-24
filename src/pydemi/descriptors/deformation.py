"""
pydemi.descriptors.deformation
------------------------------
Family A -- deformation density (reference entries 40-47):

    delta_rho(r) = rho(r) - sum_i rho^free_e(i)(|r - R_i|)

Promolecule
-----------
Built in reciprocal space: each element's radial density is Fourier-Bessel
transformed once into a form factor f_e(G) = 4 pi int n(r) r^2 j0(Gr) dr,
and

    rho_pro(G) = sum_e f_e(|G|) sum_{i in e} exp(-i G . R_i)

is summed over every G of the FFT grid. That is the exact periodic sum (no
real-space cutoff), band-limited like every VASP grid density, and its
integral is exactly sum_i N_i.

Which reference
---------------
``field="rho_ae"`` (AECCAR0 + AECCAR2) is compared with total free-atom
densities: consistent, and the route the reference recommends. This is
what ``compute_descriptors`` uses; without AECCARs it returns NaN.

``field="rho"`` (CHGCAR) must be compared with a reference carrying the
same PAW pseudization. With the tabulated all-electron valence orbitals it
is not: on fcc FeCoNiCr the pseudized CHGCAR holds 0.4 e within 0.4 A of
Fe where the free-atom valence holds 2.0 e, and ~3 e per Fe appear in the
0.8-1.5 A shell instead -- so delta_rho measures the POTCAR, not bonding,
and excluding the core shell does not help (Fe's PAW sphere reaches
~1.2 A). Use it only with a pseudized reference built from isolated-atom
CHGCARs run with the production POTCARs
(:class:`pydemi.atoms.reference.IsolatedAtomReference`), where the
pseudization cancels; with the LDA reference it is allowed only when
asked for explicitly, and warns.

``def_all_electron`` records which field was used. ``def_charge_mismatch``
= int delta_rho dV should be ~0: a large value means the reference electron
counts do not match the calculation (ZVAL: pass it or load the POTCAR).

Descriptors (moments weight by |delta_rho| so the normalization is a proper
measure; sign information is carried by the accumulation/depletion shares):

    m1_def, m2_def, sigma_r2_def        (40-42)
    f_bond_def, f_int_def               (43-44)  shares of accumulated charge
    f_bond_dep                          (45)     share of depleted charge
    bond_charge_transfer                (46)     int_bond delta_rho dV, electrons
    def_polarity                        (47)     int |delta_rho| dV / Q_tot
"""

import warnings
from typing import Optional

import numpy as np

from ..engine import RHO, RHO_AE, Engine
from ..shells import Shells
from .primitives import radial_moments, safe_div

PROMOLECULE = "promolecule"
MISMATCH_WARN = 0.1   # electrons


def form_factor(r: np.ndarray, n: np.ndarray, G: np.ndarray, chunk: int = 512) -> np.ndarray:
    """f(G) = 4 pi int n(r) r^2 sin(Gr)/(Gr) dr on a logarithmic r grid."""
    x = np.log(r)
    w = np.gradient(x) * 4.0 * np.pi * n * r ** 3            # dr = r dx
    out = np.empty(len(G))
    for s in range(0, len(G), chunk):
        Gr = np.outer(G[s:s + chunk], r)
        out[s:s + chunk] = np.sinc(Gr / np.pi) @ w
    return out


def reference_part(field: str) -> str:
    return "total" if field == RHO_AE else "valence"


def promolecule(engine: Engine, field: str = RHO, n_table: int = 4000) -> np.ndarray:
    """Promolecule density on ``field``'s grid (cached as a field)."""
    part = reference_part(field)
    name = f"{PROMOLECULE}_{part}"
    shape = engine[field].grid.shape
    if name in engine and engine[name].grid.shape == shape:
        return engine[name].values
    grid = engine.grid(shape)
    ref, zval = engine.reference, engine.valence()
    G = np.sqrt(np.maximum(grid.g_squared(), 0.0))          # rfftn half-grid, 1/Angstrom
    table = np.linspace(0.0, G.max() * 1.0001, n_table)
    m = [np.fft.fftfreq(n, 1.0 / n) for n in shape[:2]] + [np.fft.rfftfreq(shape[2], 1.0 / shape[2])]
    frac = engine.structure.frac_coords
    F = np.zeros(G.shape, dtype=complex)
    for e in engine.structure.elements:
        r, n = ref.radial(e, part, zval.get(e))
        f_e = np.interp(G, table, form_factor(r, n, table))
        S = np.zeros(G.shape, dtype=complex)
        for i in np.flatnonzero(np.array(engine.structure.species) == e):
            p = [np.exp(-2j * np.pi * mm * frac[i, ax]) for ax, mm in enumerate(m)]
            S += p[0][:, None, None] * p[1][None, :, None] * p[2][None, None, :]
        F += f_e * S
    rho = np.fft.irfftn(F * (grid.n_voxels / grid.volume), s=shape, axes=(0, 1, 2))
    engine.add_field(name, rho)
    return rho


def deformation_density(engine: Engine, field: Optional[str] = None) -> tuple:
    """(delta_rho, field used)."""
    if field is None:
        field = RHO_AE if RHO_AE in engine else RHO
    return engine[field].values - promolecule(engine, field), field


_NAMES = ("m1_def", "m2_def", "sigma_r2_def", "f_bond_def", "f_int_def", "f_bond_dep",
          "bond_charge_transfer", "def_polarity", "def_charge_mismatch")


def deformation_family(engine: Engine, field: Optional[str] = None,
                       shells: Optional[Shells] = None) -> dict:
    """Entries 40-47. ``field=None``: AECCAR if loaded, else NaN (see module docstring)."""
    if field is None:
        if RHO_AE not in engine:
            out = {k: float("nan") for k in _NAMES}
            out["def_all_electron"] = 0
            return out
        field = RHO_AE
    if field == RHO and not getattr(engine.reference, "pseudized", False):
        warnings.warn(
            "deformation density of CHGCAR against all-electron free-atom valence "
            "densities is dominated by PAW pseudization, not bonding; use AECCAR "
            "(field='rho_ae') or an IsolatedAtomReference built with the same POTCARs",
            stacklevel=2)
    drho, field = deformation_density(engine, field)
    f = engine[field]
    dV = f.grid.dV
    shape = f.grid.shape
    r = engine.geometry(shape).distance
    masks = engine.shell_masks(shape, shells)
    a = np.abs(drho)
    pos, neg = drho > 0, drho < 0
    Q_tot = f.integral()
    mismatch = float(drho.sum() * dV)
    if abs(mismatch) > MISMATCH_WARN:
        warnings.warn(
            f"deformation density integrates to {mismatch:+.3f} e: the free-atom "
            f"electron counts ({engine.valence() if field == RHO else 'Z'}) do not match "
            "the calculation's; pass zval= or load the POTCAR", stacklevel=2)
    m1, m2, var = radial_moments(a, r)
    return {
        "m1_def": m1,
        "m2_def": m2,
        "sigma_r2_def": var,
        "f_bond_def": safe_div(drho[masks.bond & pos].sum(), drho[pos].sum()),
        "f_int_def": safe_div(drho[masks.interstitial & pos].sum(), drho[pos].sum()),
        "f_bond_dep": safe_div(a[masks.bond & neg].sum(), a[neg].sum()),
        "bond_charge_transfer": float(drho[masks.bond].sum() * dV),
        "def_polarity": safe_div(a.sum() * dV, Q_tot),
        "def_charge_mismatch": mismatch,
        "def_all_electron": int(field == RHO_AE),
    }
