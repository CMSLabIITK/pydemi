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

``field="rho"`` (CHGCAR) against the tabulated all-electron valence
orbitals is inconsistent inside the PAW augmentation spheres, where CHGCAR
is pseudized. Measured on 121 VASP CHGCARs: 87% (median) of int |delta_rho|
lies inside the spheres, which fill 45% of the volume, while the charge
inside them is conserved to 0.6%; outside the spheres the two densities
agree (FeNi3: within ~3% beyond 0.8 A of Fe). So |delta_rho|-weighted
whole-cell descriptors (m1_def, def_polarity, ...) are dominated by the
pseudization, although delta_rho outside the spheres reflects bonding. The
CHGCAR route is therefore allowed only when asked for explicitly, and warns
unless the reference is pseudized in the same way
(:class:`pydemi.atoms.reference.IsolatedAtomReference`, built from
isolated-atom CHGCARs run with the production POTCARs).

``def_all_electron`` records which field was used. ``def_charge_mismatch``
= int delta_rho dV should be ~0: a large value means the reference electron
counts do not match the calculation (ZVAL: pass it or load the POTCAR).

Outside the PAW spheres (``*_def_out``, pydemi addition)
-------------------------------------------------------
Beyond the augmentation radius R_PAW of its nucleus a CHGCAR voxel is not
pseudized, so there the CHGCAR and the valence free-atom reference are
consistent. :func:`deformation_outside_paw` evaluates the same eight
quantities over those voxels only, from the CHGCAR -- no AECCARs needed.
R_PAW is RCORE from the POTCAR / OUTCAR (``def_out_radii_from_paw`` = 1) or,
when unknown, the covalent radius. The shell fractions then refer to the
part of each shell outside the spheres; ``def_out_volume_fraction`` reports
how much of the cell that is (atoms with R_PAW > c2 leave no bond shell).
Charge transfer across a sphere boundary is physical, so there is no
charge-conservation check for this variant.

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


_OUT_NAMES = ("m1_def_out", "m2_def_out", "sigma_r2_def_out", "f_bond_def_out",
              "f_int_def_out", "f_bond_dep_out", "bond_charge_transfer_out",
              "def_polarity_out")


def deformation_outside_paw(engine: Engine, shells: Optional[Shells] = None) -> dict:
    """Family A over the voxels outside every PAW augmentation sphere (CHGCAR)."""
    if RHO not in engine:
        out = {k: float("nan") for k in _OUT_NAMES}
        out.update(def_out_volume_fraction=float("nan"), def_out_radii_from_paw=0)
        return out
    f = engine[RHO]
    shape = f.grid.shape
    shells = shells if shells is not None else engine.shells
    drho = f.values - promolecule(engine, RHO)
    geo = engine.geometry(shape)
    R, known = engine.augmentation_radii(fallback=shells.c1)
    out_mask = geo.distance > R[geo.atom_index]
    masks = engine.shell_masks(shape, shells)
    dV = f.grid.dV
    d = np.where(out_mask, drho, 0.0)
    a = np.abs(d)
    pos, neg = d > 0, d < 0
    m1, m2, var = radial_moments(a, geo.distance)
    return {
        "m1_def_out": m1,
        "m2_def_out": m2,
        "sigma_r2_def_out": var,
        "f_bond_def_out": safe_div(d[masks.bond & pos].sum(), d[pos].sum()),
        "f_int_def_out": safe_div(d[masks.interstitial & pos].sum(), d[pos].sum()),
        "f_bond_dep_out": safe_div(a[masks.bond & neg].sum(), a[neg].sum()),
        "bond_charge_transfer_out": float(d[masks.bond].sum() * dV),
        "def_polarity_out": safe_div(a.sum() * dV, f.integral()),
        "def_out_volume_fraction": float(out_mask.mean()),
        "def_out_radii_from_paw": int(known),
    }
