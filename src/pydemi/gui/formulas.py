"""Typeset (LaTeX) formulas of the descriptors, for display in the GUI.

The catalogue keeps a plain-text formula per descriptor (``pydemi.catalogue()``, also in
the CSV exports); these are the same definitions written for KaTeX, in the notation of
the descriptor reference (docs/descriptors.tex). ``FORMULAS[name]`` is (main formula,
lines that define its symbols): markdown with inline math, or, between ``$$``, a
displayed equation.
"""

from __future__ import annotations

# shared definitions
_SHELLS = r"Shells around the nearest nucleus: core $r_k \le c_1$, bond $c_1 < r_k \le c_2$, interstitial $r_k > c_2$ (defaults $c_1 = 0.8$ Å, $c_2 = 1.5$ Å)."
_RK = r"$\rho_k$: density at voxel $k$; $r_k$: distance to its nearest nucleus; sums run over the $N$ voxels of the cell."
_UK = r"$\hat u_k$: unit vector from the nearest nucleus to voxel $k$."
_LAP = r"$\nabla^2\rho_k$: Laplacian of the density at voxel $k$ (spectral derivative)."
_NCI = (r"NCI region: voxels with $s_k < 0.5$ and $\rho_k < 0.05$ a.u., where "
        r"$s = |\nabla\rho| \,/\, \big(2(3\pi^2)^{1/3}\rho^{4/3}\big)$; $\lambda_2$: middle Hessian eigenvalue.")
_T = r"$$T_{ab} = \frac{\sum_k \partial_a\rho_k\,\partial_b\rho_k}{\sum_k |\nabla\rho_k|^2}$$"
_TILDE = r"$\tilde\rho = \rho / N_e$: the density normalised to one electron; $V$: cell volume."
_PERC = r"$\{\rho > c\}$: the region where the density exceeds $c$; it spans $\mathbf{a}_\alpha$ if it connects a cell to its periodic image along $\mathbf{a}_\alpha$."
_SITE = r"$w_i(k) = 1$ if voxel $k$ belongs to atom $i$ (the chosen partition), else 0; $r_{ik}$ and $\hat u_{ik}$: distance and direction from atom $i$."
_ELEMENT = r"$x_e$: fraction of the sites that are element $e$; $\bar X_e$: mean over the sites of $e$; $\bar X$: mean over all sites."

_SITE_X = {
    "m1": (r"m_1", r"m_1^{(i)} = \frac{\sum_k w_i(k)\,\rho_k\, r_{ik}}{\sum_k w_i(k)\,\rho_k}"),
    "f_bond": (r"f_{\mathrm{bond}}",
               r"f_{\mathrm{bond}}^{(i)} = \frac{\sum_{k:\, c_1 < r_{ik} \le c_2} w_i(k)\,\rho_k}{\sum_k w_i(k)\,\rho_k}"),
    "zeta": (r"\zeta", r"\zeta^{(i)} = 1 - \frac{\sum_k w_i(k)\,|\nabla\rho_k \cdot \hat u_{ik}|}{\sum_k w_i(k)\,|\nabla\rho_k|}"),
}
_SITE_STATS = {
    "site_std": r"\operatorname{std}_i\, {X}^{(i)}",
    "site_range": r"\max_i {X}^{(i)} - \min_i {X}^{(i)}",
    "site_max": r"\max_i {X}^{(i)}",
    "site_min": r"\min_i {X}^{(i)}",
    "within_element_var": r"\sum_e x_e \operatorname{Var}_{i \in e}\, {X}^{(i)}",
    "between_element_var": r"\sum_e x_e \left(\bar{X}_e - \bar{X}\right)^2",
}

FORMULAS: dict[str, tuple[str, list[str]]] = {
    # --- bonding -----------------------------------------------------------
    "zeta": (r"\zeta = 1 - \frac{\sum_k |\nabla\rho_k \cdot \hat u_k|}{\sum_k |\nabla\rho_k|}", [_UK]),
    "m1": (r"m_1 = \frac{\sum_k \rho_k\, r_k}{\sum_k \rho_k}", [_RK]),
    "m2": (r"m_2 = \frac{\sum_k \rho_k\, r_k^2}{\sum_k \rho_k}", [_RK]),
    "sigma_r2": (r"\sigma_r^2 = m_2 - m_1^2", [r"$m_1$, $m_2$: first and second radial moments of the charge."]),
    "f_core": (r"f_{\mathrm{core}} = \frac{\sum_{k:\, r_k \le c_1} \rho_k}{\sum_k \rho_k}", [_SHELLS]),
    "f_bond": (r"f_{\mathrm{bond}} = \frac{\sum_{k:\, c_1 < r_k \le c_2} \rho_k}{\sum_k \rho_k}", [_SHELLS]),
    "f_int": (r"f_{\mathrm{int}} = \frac{\sum_{k:\, r_k > c_2} \rho_k}{\sum_k \rho_k}", [_SHELLS]),
    "lnf": (r"\mathrm{lnf} = \frac{1}{N} \sum_k \mathbf{1}\!\left[\nabla^2\rho_k < 0\right]",
            [_LAP, r"$\mathbf{1}[\cdot]$ is 1 where the condition holds, else 0."]),
    "lnf_charge_weighted": (r"\mathrm{lnf}_\rho = \frac{\sum_{k:\, \nabla^2\rho_k < 0} \rho_k}{\sum_k \rho_k}", [_LAP]),
    "lap_concentration": (r"\frac{\sum_{k:\, \nabla^2\rho_k < 0} |\nabla^2\rho_k|}{\sum_k |\nabla^2\rho_k|} = \frac{1}{2}",
                          [_LAP, r"On a periodic grid $\int \nabla^2\rho \, dV = 0$, so this ratio is always $1/2$."]),
    "lap_concentration_valence": (
        r"\frac{\sum_{k:\, r_k > c_1,\ \nabla^2\rho_k < 0} |\nabla^2\rho_k|}{\sum_{k:\, r_k > c_1} |\nabla^2\rho_k|}",
        [_LAP, r"Only voxels outside the cores ($r_k > c_1$)."]),
    "ellip_bond_bounded_avg": (r"\left\langle 1 - \frac{\lambda_2}{\lambda_1} \right\rangle_{k \in B} = \left\langle \frac{\varepsilon}{1 + \varepsilon} \right\rangle_{k \in B}",
                               [r"$B = \{k \in \text{bond shell} : \lambda_2 < 0\}$; $\lambda_1 \le \lambda_2$: the two lowest Hessian eigenvalues; $\varepsilon = \lambda_1/\lambda_2 - 1$ is the usual ellipticity."]),
    "f_NCI": (r"f_{\mathrm{NCI}} = \frac{1}{N} \sum_k \mathbf{1}\!\left[k \in \mathrm{NCI}\right]", [_NCI]),
    "NCI_attractive": (r"\frac{\#\{k \in \mathrm{NCI} : \lambda_2 < 0\}}{\#\{k \in \mathrm{NCI}\}}", [_NCI]),
    "sign_lambda2_rho_mean": (r"\left\langle \operatorname{sign}(\lambda_2)\, \rho_k \right\rangle_{k \in \mathrm{NCI}}",
                              [_NCI, r"$\rho$ in atomic units (e/bohr³)."]),
    "V_spread": (r"\operatorname{std}_i\, V(\mathbf{R}_i)",
                 [r"$V$: Hartree potential of the density (cell average set to 0); $\mathbf{R}_i$: position of atom $i$."]),
    "V_int_min": (r"\min_{k:\, r_k > c_2} V_k",
                  [r"$V$: Hartree potential of the density, cell average set to 0; the minimum over the interstitial voxels."]),
    "rho_mid_mean": (r"\left\langle \rho\!\left(\tfrac{\mathbf{R}_i + \mathbf{R}_j}{2}\right) \right\rangle_{(i,j)}",
                     [r"$(i, j)$: nearest-neighbour pairs of atoms; the density is read at the midpoint of each bond."]),
    "rho_mid_std": (r"\operatorname{std}_{(i,j)}\; \rho\!\left(\tfrac{\mathbf{R}_i + \mathbf{R}_j}{2}\right)",
                    [r"$(i, j)$: nearest-neighbour pairs of atoms; the density is read at the midpoint of each bond."]),
    # --- structural --------------------------------------------------------
    "rho_perc_a": (r"\rho_{\mathrm{perc},a} = \max\left\{ c : \{\rho > c\} \text{ spans } \mathbf{a}_1 \right\}", [_PERC]),
    "rho_perc_b": (r"\rho_{\mathrm{perc},b} = \max\left\{ c : \{\rho > c\} \text{ spans } \mathbf{a}_2 \right\}", [_PERC]),
    "rho_perc_c": (r"\rho_{\mathrm{perc},c} = \max\left\{ c : \{\rho > c\} \text{ spans } \mathbf{a}_3 \right\}", [_PERC]),
    "perc_anisotropy": (r"\frac{\max_\alpha \rho_{\mathrm{perc},\alpha} - \min_\alpha \rho_{\mathrm{perc},\alpha}}{\operatorname{mean}_\alpha \rho_{\mathrm{perc},\alpha}}",
                        [r"$\alpha \in \{a, b, c\}$: the three lattice directions."]),
    "n_max": (r"n_{\max} = \frac{\#\{\text{local maxima of } \rho\}}{V}",
              [r"A local maximum is a voxel higher than all 26 neighbours; $V$: cell volume."]),
    "n_min": (r"n_{\min} = \frac{\#\{\text{local minima of } \rho\}}{V}",
              [r"A local minimum is a voxel lower than all 26 neighbours; $V$: cell volume."]),
    "n_NNM": (r"n_{\mathrm{NNM}} = \frac{\#\{\text{maxima with } \min_i |\mathbf{r} - \mathbf{R}_i| > r_{\mathrm{cut}}\}}{V}",
              [r"Non-nuclear maxima: local maxima farther than $r_{\mathrm{cut}} = 0.8$ Å from every nucleus."]),
    "Q_NNM": (r"Q_{\mathrm{NNM}} = \frac{1}{Q_{\mathrm{tot}}} \sum_{k \in \Omega_{\mathrm{NNM}}} \rho_k\, \Delta V",
              [r"$\Omega_{\mathrm{NNM}}$: the steepest-ascent basins of the non-nuclear maxima; $Q_{\mathrm{tot}} = \sum_k \rho_k\,\Delta V$."]),
    "rho_min": (r"\rho_{\min} = \min_k \rho_k", [r"Often slightly negative in a PAW pseudo-density, near a nucleus."]),
    "rho_min_ratio": (r"\frac{\rho_{\min}}{\langle \rho \rangle_V}", [r"$\langle\rho\rangle_V = Q_{\mathrm{tot}}/V$: the mean density."]),
    "rho_int_mean": (r"\langle \rho_k \rangle_{k:\, r_k > c_2}", [_SHELLS]),
    "T_eigenvalues_t1": (r"t_1 = \text{smallest eigenvalue of } T", [_T, r"$T$: the $3\times3$ charge-anisotropy tensor; $t_1 \le t_2 \le t_3$, $t_1 + t_2 + t_3 = 1$."]),
    "T_eigenvalues_t2": (r"t_2 = \text{middle eigenvalue of } T", [_T, r"$T$: the $3\times3$ charge-anisotropy tensor; $t_1 \le t_2 \le t_3$, $t_1 + t_2 + t_3 = 1$."]),
    "T_eigenvalues_t3": (r"t_3 = \text{largest eigenvalue of } T", [_T, r"$T$: the $3\times3$ charge-anisotropy tensor; $t_1 \le t_2 \le t_3$, $t_1 + t_2 + t_3 = 1$."]),
    "charge_FA": (r"\mathrm{FA} = \sqrt{\tfrac{3}{2}}\; \frac{\lVert T - \tfrac{1}{3} I \rVert_F}{\lVert T \rVert_F}",
                  [_T, r"$T$: the charge-anisotropy tensor; $\lVert\cdot\rVert_F$: Frobenius norm; $I$: identity."]),
    "shannon_entropy": (r"S = -\int \tilde\rho \ln \tilde\rho \; dV - \ln V", [_TILDE]),
    "fisher_information": (r"I = \int \frac{|\nabla \tilde\rho|^2}{\tilde\rho} \; dV",
                           [_TILDE, r"In atomic units, over the voxels above the density floor."]),
    "disequilibrium": (r"D = V \int \tilde\rho^{\,2} \; dV", [_TILDE]),
    "LMC_complexity": (r"C = D\, e^{S}", [r"$D$, $S$: the disequilibrium and Shannon entropy above; $C = 1$ for a uniform density."]),
}

for _x, (_sym, _def) in _SITE_X.items():
    for _stat, _tex in _SITE_STATS.items():
        FORMULAS[f"{_x}_{_stat}"] = (_tex.replace("{X}", "{" + _sym + "}").replace(r"\bar{X}", r"\bar{" + _sym + "}"),
                                     [f"$${_def}$$", _SITE] + ([_ELEMENT] if "element" in _stat else []))


def formula(name: str) -> tuple[str, list[str]]:
    """(LaTeX formula, markdown definitions) of descriptor ``name``."""
    return FORMULAS[name]
