"""
pydemi.grid
-----------
Periodic real-space grid and derivative operators.

A field f is sampled at voxel (i, j, k) <-> fractional position
u = (i/n1, j/n2, k/n3), Cartesian x = u @ A where A = lattice (rows a_i).

Derivatives
-----------
Two methods, both periodic (one-sided differences on the boundary planes,
as ``np.gradient`` uses, are wrong for a periodic cell):

``"fd"`` (default)
    Second-order central differences with compact stencils, via np.roll.
    Index-space derivatives are mapped to Cartesian with the chain rule:

        grad_x = A^{-1} (n * d_i f)
        H_x    = A^{-1} H_u A^{-T},  H_u[m, n] = n_m n_n d^2 f / di_m di_n

    so the Laplacian tr(H_x) = sum_mn G^{mn} H_u[m, n] with the
    contravariant metric G = A^{-T} A^{-1} -- exact for any cell shape,
    including the cross-axis terms a diagonal approximation drops.

``"spectral"``
    Exact derivatives of the trigonometric interpolant, via FFT:
    d_a -> i G_a, d_a d_b -> -G_a G_b. VASP grids are FFT grids, so this
    is the natural exact choice, but it amplifies high-frequency content
    (e.g. near PAW cores) more than FD does. Nyquist modes are dropped
    from odd-order terms so the result is real and symmetric.

The default is "fd". Switching method is a sensitivity axis, not a free
choice, for sign-thresholded descriptors such as lnf. In particular the
spectral Laplacian has an absolute round-off floor (~1e-16 of its peak)
over the whole cell, so in near-empty regions (vacuum, molecular-crystal
voids) its sign is noise and voxel-count fractions become unreliable;
FD stencils stay accurate relative to the local density there.
"""

from functools import cached_property

import numpy as np

_METHODS = ("fd", "spectral")

# packed symmetric 3x3 storage order
HESSIAN_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


class Grid:
    """Geometry of a periodic grid of ``shape`` voxels spanning ``lattice``."""

    def __init__(self, lattice, shape):
        self.lattice = np.asarray(lattice, dtype=float).reshape(3, 3)
        self.shape = tuple(int(n) for n in shape)
        if len(self.shape) != 3 or min(self.shape) < 1:
            raise ValueError(f"bad grid shape {shape}")

    def __repr__(self):
        return f"Grid(shape={self.shape}, dV={self.dV:.3e} A^3)"

    # ---------------- geometry ----------------

    @cached_property
    def inv_lattice(self) -> np.ndarray:
        return np.linalg.inv(self.lattice)

    @cached_property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.lattice)))

    @property
    def n_voxels(self) -> int:
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def dV(self) -> float:
        return self.volume / self.n_voxels

    @cached_property
    def spacing(self) -> np.ndarray:
        """Voxel edge length along each lattice vector (Angstrom)."""
        return np.linalg.norm(self.lattice, axis=1) / np.array(self.shape)

    @cached_property
    def metric(self) -> np.ndarray:
        """Contravariant metric G = A^{-T} A^{-1} (1/Angstrom^2)."""
        return self.inv_lattice.T @ self.inv_lattice

    def frac_coords(self) -> np.ndarray:
        """(nx, ny, nz, 3) fractional voxel coordinates (not cached)."""
        axes = [np.arange(n) / n for n in self.shape]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)

    def cart_coords(self) -> np.ndarray:
        """(nx, ny, nz, 3) Cartesian voxel coordinates, Angstrom (not cached)."""
        return self.frac_coords() @ self.lattice

    def integrate(self, f) -> float:
        """Integral of f over the cell: sum_k f_k dV."""
        return float(np.sum(f) * self.dV)

    # ---------------- spectral helpers ----------------

    def _frequencies(self):
        """Integer frequencies on the rfftn half-grid, as broadcastable arrays.

        Returns (m_full, m_odd): per-axis frequency arrays; m_odd has the
        Nyquist frequency of every even-length axis zeroed, for use in
        odd-order derivative terms.
        """
        m_full, m_odd = [], []
        for ax, n in enumerate(self.shape):
            m = np.fft.rfftfreq(n, 1.0 / n) if ax == 2 else np.fft.fftfreq(n, 1.0 / n)
            m_o = m.copy()
            if n % 2 == 0:
                m_o[np.isclose(np.abs(m_o), n / 2)] = 0.0
            shape = [1, 1, 1]
            shape[ax] = m.size
            m_full.append(m.reshape(shape))
            m_odd.append(m_o.reshape(shape))
        return m_full, m_odd

    @property
    def _B(self) -> np.ndarray:
        """G_a = sum_j B[a, j] m_j for integer frequencies m_j."""
        return 2.0 * np.pi * self.inv_lattice

    def _second_order_multiplier(self, W):
        """sum_jq W[j, q] m_j m_q, with Nyquist dropped from the j != q terms."""
        m_full, m_odd = self._frequencies()
        mult = 0.0
        for j in range(3):
            for q in range(3):
                if W[j, q] != 0.0:
                    mm = m_full[j] ** 2 if j == q else m_odd[j] * m_odd[q]
                    mult = mult + W[j, q] * mm
        return mult

    def g_squared(self) -> np.ndarray:
        """|G|^2 on the rfftn half-grid (1/Angstrom^2), consistent with laplacian()."""
        return self._second_order_multiplier(self._B.T @ self._B)

    def frequencies(self):
        """Integer FFT frequencies along each axis (full, not half-grid)."""
        return [np.fft.fftfreq(n, 1.0 / n) for n in self.shape]

    # ---------------- derivatives ----------------

    def _check(self, f, method):
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
        f = np.asarray(f, dtype=float)
        if f.shape != self.shape:
            raise ValueError(f"field shape {f.shape} != grid shape {self.shape}")
        return f

    def gradient(self, f, method: str = "fd") -> np.ndarray:
        """Cartesian gradient, shape (nx, ny, nz, 3), units [f]/Angstrom."""
        f = self._check(f, method)
        out = np.empty(self.shape + (3,))
        if method == "spectral":
            F = np.fft.rfftn(f)
            _, m_odd = self._frequencies()
            B = self._B
            for a in range(3):
                G_a = B[a, 0] * m_odd[0] + B[a, 1] * m_odd[1] + B[a, 2] * m_odd[2]
                out[..., a] = np.fft.irfftn(1j * G_a * F, s=self.shape, axes=(0, 1, 2))
            return out

        n = np.array(self.shape, dtype=float)
        d_u = [n[m] * 0.5 * (np.roll(f, -1, m) - np.roll(f, 1, m)) for m in range(3)]
        Ainv = self.inv_lattice
        for a in range(3):
            out[..., a] = Ainv[a, 0] * d_u[0] + Ainv[a, 1] * d_u[1] + Ainv[a, 2] * d_u[2]
        return out

    def _hessian_frac(self, f):
        """Index->fractional second derivatives H_u[m, n], packed by HESSIAN_PAIRS."""
        n = np.array(self.shape, dtype=float)
        H_u = {}
        for m in range(3):
            H_u[(m, m)] = n[m] ** 2 * (np.roll(f, -1, m) - 2.0 * f + np.roll(f, 1, m))
        for m, q in ((0, 1), (0, 2), (1, 2)):
            fp = np.roll(f, -1, m)
            fm = np.roll(f, 1, m)
            mixed = 0.25 * (np.roll(fp, -1, q) - np.roll(fp, 1, q)
                            - np.roll(fm, -1, q) + np.roll(fm, 1, q))
            H_u[(m, q)] = n[m] * n[q] * mixed
        return H_u

    def hessian(self, f, method: str = "fd") -> np.ndarray:
        """Cartesian Hessian packed as (nx, ny, nz, 6) in HESSIAN_PAIRS order.

        Use :func:`unpack_hessian` for a full (..., 3, 3) view.
        """
        f = self._check(f, method)
        out = np.empty(self.shape + (6,))
        if method == "spectral":
            F = np.fft.rfftn(f)
            B = self._B
            for p, (a, b) in enumerate(HESSIAN_PAIRS):
                W = 0.5 * (np.outer(B[a], B[b]) + np.outer(B[b], B[a]))
                mult = self._second_order_multiplier(W)
                out[..., p] = np.fft.irfftn(-mult * F, s=self.shape, axes=(0, 1, 2))
            return out

        H_u = self._hessian_frac(f)
        Ainv = self.inv_lattice
        for p, (a, b) in enumerate(HESSIAN_PAIRS):
            acc = np.zeros(self.shape)
            for m in range(3):
                for q in range(3):
                    c = Ainv[a, m] * Ainv[b, q]
                    if c != 0.0:
                        acc += c * H_u[(min(m, q), max(m, q))]
            out[..., p] = acc
        return out

    def laplacian(self, f, method: str = "fd") -> np.ndarray:
        """Cartesian Laplacian, shape (nx, ny, nz), units [f]/Angstrom^2."""
        f = self._check(f, method)
        if method == "spectral":
            F = np.fft.rfftn(f)
            return np.fft.irfftn(-self.g_squared() * F, s=self.shape, axes=(0, 1, 2))

        H_u = self._hessian_frac(f)
        G = self.metric
        lap = np.zeros(self.shape)
        for (m, q), h in H_u.items():
            lap += (G[m, q] if m == q else 2.0 * G[m, q]) * h
        return lap


def unpack_hessian(packed: np.ndarray) -> np.ndarray:
    """(..., 6) packed symmetric -> (..., 3, 3) full."""
    full = np.empty(packed.shape[:-1] + (3, 3), dtype=packed.dtype)
    for p, (a, b) in enumerate(HESSIAN_PAIRS):
        full[..., a, b] = packed[..., p]
        full[..., b, a] = packed[..., p]
    return full


def hessian_eigenvalues(packed: np.ndarray, chunk: int = 1 << 20) -> np.ndarray:
    """Ascending eigenvalues lambda1 <= lambda2 <= lambda3 of a packed Hessian.

    Processed in chunks so the full (..., 3, 3) array is never materialized.
    """
    flat = packed.reshape(-1, 6)
    out = np.empty((flat.shape[0], 3))
    for s in range(0, flat.shape[0], chunk):
        out[s:s + chunk] = np.linalg.eigvalsh(unpack_hessian(flat[s:s + chunk]))
    return out.reshape(packed.shape[:-1] + (3,))
