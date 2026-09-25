"""
pydemi.calibration
------------------
Family D (reference entries 60-62) and the rho-based bulk-modulus baseline
(entry 106): quantities fitted once against literature values.

grid_ionicity (60)
    f = sigmoid(b0 + sum_j b_j z_j), z_j the standardized descriptors in
    ``features``, fitted by least squares to Phillips f_i on reference
    compounds whose densities you have computed. The sigmoid keeps the
    prediction on Phillips' [0, 1] scale. The default features pair
    ``fint_over_lnf`` with the electrostatic site-potential spread, as the
    reference recommends, so the calibration does not rest only on the
    descriptor the SHAP analysis singled out (use ``V_spread`` from a
    LOCPOT when available; ``VH_spread`` is the electronic Hartree
    stand-in). Leave-one-out RMSE is reported with every fit. Phillips'
    scale was built for tetrahedral semiconductors: outside that class the
    prediction is an extrapolation, and the fit reports how many of its
    compounds were tetrahedral.

cohen_B0_predicted (61)
    Cohen, Phys. Rev. B 32, 7988 (1985): B0 = (1971 - 220 lambda) d^-3.5 GPa,
    d the nearest-neighbour distance in Angstrom. Cohen's lambda is an
    integer ionicity *class* -- 0 for group IV, 1 for III-V, 2 for II-VI --
    not a continuous ionicity, so it is taken from the composition. (The
    descriptor reference writes 1972 and lambda = grid_ionicity; both are
    departures from Cohen's formula, whose [0, 2] class scale a [0, 1]
    ionicity cannot stand in for.) ``cohen_in_scope`` is 1 only for
    those classes; elsewhere lambda = 0 (the covalent form) is used and the
    value is a formula extrapolation, not a physics baseline. d is the
    per-atom first-shell mean (``bond_length_mean``).

ionicity_residual (62)
    grid_ionicity - ionicity (the Pauling compositional estimate, entry 26).

B0_rho_proxy (106)
    B0 = a x^b with x = rho_mid_mean / bond_length_mean^3, fitted in log space
    on structures with known bulk moduli.
"""

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

_DATA = Path(__file__).resolve().parent.parent / "data" / "phillips_ionicity.csv"
DEFAULT_IONICITY_FEATURES = ("fint_over_lnf", "VH_spread")
COHEN_A, COHEN_B = 1971.0, 220.0


# ---------------------------------------------------------------- literature

def phillips_table(verified_only: bool = False) -> dict:
    """formula -> row dict (f_i, structure, cohen_lambda, status, source)."""
    lines = [l for l in _DATA.read_text().splitlines() if not l.startswith("#")]
    out = {}
    for row in csv.DictReader(lines):
        if verified_only and row["status"] != "verified":
            continue
        row["f_i"] = float(row["f_i"])
        row["cohen_lambda"] = int(row["cohen_lambda"]) if row["cohen_lambda"] else None
        out[row["formula"]] = row
    return out


def reduced_formula(species) -> str:
    """Reduced formula in Phillips-table style (cation first): GaAs, NaCl, SiC."""
    from pymatgen.core import Composition
    return Composition("".join(species)).reduced_formula


# ---------------------------------------------------------------- Cohen (61)

def cohen_lambda(species) -> Optional[int]:
    """Cohen's ionicity class from composition: 0 IV, 1 III-V, 2 II-VI, else None."""
    from pymatgen.core import Composition, Element
    comp = Composition("".join(species)).reduced_composition
    els = [Element(str(e)) for e in comp.elements]
    groups = sorted(e.group for e in els)
    if len(els) == 1 and groups == [14]:
        return 0
    if len(els) != 2 or any(abs(comp[e] - 1.0) > 1e-9 for e in comp.elements):
        return None
    if groups == [14, 14]:
        return 0
    if groups == [13, 15]:
        return 1
    if groups in ([2, 16], [12, 16]):
        return 2
    return None


def cohen_bulk_modulus(d: float, lam: float) -> float:
    """Cohen's B0 in GPa for nearest-neighbour distance d (Angstrom)."""
    return (COHEN_A - COHEN_B * lam) * d ** -3.5


# ---------------------------------------------------------------- ionicity (60)

def _sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


@dataclass
class IonicityCalibration:
    features: tuple
    mean: list
    scale: list
    coef: list                  # [b0, b1, ...]
    n: int
    rmse: float
    loocv_rmse: float
    n_tetrahedral: int = 0
    compounds: list = field(default_factory=list)

    def predict(self, descriptors: Mapping[str, float]) -> float:
        x = np.array([float(descriptors.get(f, np.nan)) for f in self.features])
        if not np.all(np.isfinite(x)):
            return float("nan")
        z = (x - np.array(self.mean)) / np.array(self.scale)
        return float(_sigmoid(self.coef[0] + z @ np.array(self.coef[1:])))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path) -> "IonicityCalibration":
        d = json.loads(Path(path).read_text())
        d["features"] = tuple(d["features"])
        return cls(**d)


def _fit_sigmoid(Z, y):
    def resid(b):
        return _sigmoid(b[0] + Z @ b[1:]) - y
    return least_squares(resid, np.zeros(Z.shape[1] + 1)).x


def fit_ionicity(descriptor_rows: Mapping[str, Mapping[str, float]],
                 features: Sequence[str] = DEFAULT_IONICITY_FEATURES,
                 targets: Optional[Mapping[str, float]] = None,
                 verified_only: bool = False) -> IonicityCalibration:
    """Fit grid_ionicity on reference compounds.

    ``descriptor_rows`` maps a formula (e.g. "GaAs") to that compound's
    descriptor dict. Targets default to the Phillips table; compounds
    without a target or with a non-finite feature are skipped.
    """
    table = phillips_table(verified_only)
    targets = dict(targets) if targets is not None else {k: v["f_i"] for k, v in table.items()}
    used, X, y = [], [], []
    for formula, desc in descriptor_rows.items():
        if formula not in targets:
            continue
        x = [float(desc.get(f, np.nan)) for f in features]
        if np.all(np.isfinite(x)):
            used.append(formula)
            X.append(x)
            y.append(targets[formula])
    X, y = np.array(X), np.array(y)
    if len(y) < len(features) + 2:
        raise ValueError(f"need at least {len(features) + 2} reference compounds, got {len(y)}")
    mean, scale = X.mean(0), X.std(0)
    scale[scale == 0] = 1.0
    Z = (X - mean) / scale
    b = _fit_sigmoid(Z, y)
    rmse = float(np.sqrt(np.mean((_sigmoid(b[0] + Z @ b[1:]) - y) ** 2)))
    errs = []
    for k in range(len(y)):                       # leave-one-out
        keep = np.arange(len(y)) != k
        bk = _fit_sigmoid(Z[keep], y[keep])
        errs.append(_sigmoid(bk[0] + Z[k] @ bk[1:]) - y[k])
    n_tet = sum(1 for f in used if f in table and table[f]["structure"]
                in ("diamond", "zincblende", "wurtzite"))
    return IonicityCalibration(tuple(features), mean.tolist(), scale.tolist(), b.tolist(),
                               len(y), rmse, float(np.sqrt(np.mean(np.square(errs)))),
                               n_tet, used)


# ---------------------------------------------------------------- B0 proxy (106)

@dataclass
class BulkModulusCalibration:
    a: float
    b: float
    n: int
    rmse_log: float

    def predict(self, descriptors: Mapping[str, float]) -> float:
        x = bulk_proxy_x(descriptors)
        return float(self.a * x ** self.b) if np.isfinite(x) and x > 0 else float("nan")

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path) -> "BulkModulusCalibration":
        return cls(**json.loads(Path(path).read_text()))


def bulk_proxy_x(descriptors: Mapping[str, float]) -> float:
    rho, d = descriptors.get("rho_mid_mean", np.nan), descriptors.get("bond_length_mean", np.nan)
    return float(rho / d ** 3) if np.isfinite(rho) and np.isfinite(d) and d > 0 else float("nan")


def fit_bulk_modulus(descriptor_rows: Sequence[Mapping[str, float]],
                     B0: Sequence[float]) -> BulkModulusCalibration:
    """Fit log B0 = log a + b log x on structures with known bulk moduli (GPa)."""
    x = np.array([bulk_proxy_x(r) for r in descriptor_rows])
    B0 = np.asarray(B0, float)
    keep = np.isfinite(x) & (x > 0) & np.isfinite(B0) & (B0 > 0)
    if keep.sum() < 3:
        raise ValueError("need at least 3 structures with finite proxy and B0")
    b, loga = np.polyfit(np.log(x[keep]), np.log(B0[keep]), 1)
    resid = np.log(B0[keep]) - (loga + b * np.log(x[keep]))
    return BulkModulusCalibration(float(np.exp(loga)), float(b), int(keep.sum()),
                                  float(np.sqrt(np.mean(resid ** 2))))


# ---------------------------------------------------------------- family D

def calibration_family(structure, descriptors: Mapping[str, float],
                       ionicity: Optional[IonicityCalibration] = None,
                       bulk: Optional[BulkModulusCalibration] = None) -> dict:
    """Entries 60-62 and 106 from already-computed descriptors."""
    lam = cohen_lambda(structure.species)
    d = descriptors.get("bond_length_mean", np.nan)
    gi = ionicity.predict(descriptors) if ionicity is not None else float("nan")
    return {
        "grid_ionicity": gi,
        "ionicity_residual": gi - descriptors.get("ionicity", np.nan),
        "cohen_B0_predicted": cohen_bulk_modulus(d, 0 if lam is None else lam)
        if np.isfinite(d) else float("nan"),
        "cohen_in_scope": int(lam is not None),
        "B0_rho_proxy": bulk.predict(descriptors) if bulk is not None else float("nan"),
    }
