"""Ellipticity statistics over bond-shell voxels: mean / std (the spec descriptors) vs robust forms,
across FFT, FD4, FD2 derivatives and a Fourier-coarsened (80%) grid, for 48 random structures (seed 5).

Usage:  python i_ellipticity_variants.py
Output: out/ellipticity_variants.csv
"""
import sys, numpy as np, pandas as pd, pydemi
from concurrent.futures import ProcessPoolExecutor
from pydemi.descriptors.registry import make_options, masks
from pydemi.fields.density import derivatives
from pydemi.validate.convergence import _resampled
from common import load, sample_ids

def variants(vd, backend, order):
    ev = derivatives(vd, "rho", lambda v: v.rho.data, backend, order).hessian_eigenvalues
    sel = masks(vd.with_options(make_options())).bond & (ev[..., 1] < 0)
    l1, l2, rho = ev[..., 0][sel], ev[..., 1][sel], vd.rho.data[sel]
    e = l1 / l2 - 1.0
    ok = l2 < -0.05 * np.abs(l1)            # lambda2 at least 5% of lambda1
    return {"mean": e.mean(), "std": e.std(), "median": np.median(e),
            "rho_weighted": np.sum(rho * e) / np.sum(rho),
            "trimmed_mean": e[ok].mean(), "eps_bounded_mean": np.mean(1 - l2 / l1)}

def one(mid):
    vd = load(mid); row = {"id": mid}
    for tag, b, o in (("fft", "fft", 4), ("fd4", "fd", 4), ("fd2", "fd", 2)):
        row.update({f"{k}@{tag}": v for k, v in variants(vd, b, o).items()})
    vc = _resampled(vd, [max(4, int(round(n * 0.8))) for n in vd.shape])
    row.update({f"{k}@fft_x0.8": v for k, v in variants(vc, "fft", 4).items()})
    return row

if __name__ == "__main__":
    with ProcessPoolExecutor(16) as ex:
        df = pd.DataFrame(list(ex.map(one, sample_ids(48, seed=5), chunksize=1)))
    df.to_csv("out/ellipticity_variants.csv", index=False)
    rel = lambda a, b: ((df[a] - df[b]).abs() / np.maximum(df[a].abs(), df[b].abs())).median()
    for k in ("mean", "std", "median", "rho_weighted", "trimmed_mean", "eps_bounded_mean"):
        print(f"{k:17s} fft~fd4 {rel(k+'@fft', k+'@fd4'):.1e}  fft~fd2 {rel(k+'@fft', k+'@fd2'):.1e}  "
              f"grid x0.8 {rel(k+'@fft', k+'@fft_x0.8'):.1e}  spearman fft/fd2 {df[k+'@fft'].rank().corr(df[k+'@fd2'].rank()):.3f}  "
              f"spearman fft/x0.8 {df[k+'@fft'].rank().corr(df[k+'@fft_x0.8'].rank()):.3f}")
