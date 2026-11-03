"""Descriptive names of the descriptors, for display in the GUI.

The package, the command line and the CSV/JSON exports use the short names of
``pydemi.descriptor_names()``; the GUI shows these readable names and keeps the short name
in the property panel and the exports. The names follow Tables 1 and 2 of the pydemi paper.
"""

from __future__ import annotations

NAMES: dict[str, str] = {
    # bonding
    "zeta": "Non-radial gradient share",
    "m1": "Mean charge distance",
    "m2": "Mean squared charge distance",
    "sigma_r2": "Charge-distance variance",
    "f_core": "Core charge fraction",
    "f_bond": "Bond-shell charge fraction",
    "f_int": "Interstitial charge fraction",
    "lnf": "Negative-Laplacian volume fraction",
    "lnf_charge_weighted": "Negative-Laplacian charge fraction",
    "lap_concentration_valence": "Valence Laplacian concentration",
    "ellip_bond_bounded_avg": "Mean bond ellipticity",
    "f_NCI": "NCI volume fraction",
    "NCI_attractive": "Attractive NCI fraction",
    "sign_lambda2_rho_mean": "Signed NCI density",
    "V_spread": "Site potential spread",
    "V_int_min": "Interstitial potential minimum",
    "rho_mid_mean": "Mean bond-midpoint density",
    "rho_mid_std": "Bond-midpoint density spread",
    # structural
    "rho_perc_a": "Percolation density along a",
    "rho_perc_b": "Percolation density along b",
    "rho_perc_c": "Percolation density along c",
    "perc_anisotropy": "Percolation anisotropy",
    "n_max": "Density maxima per volume",
    "n_min": "Density minima per volume",
    "n_NNM": "Non-nuclear maxima per volume",
    "Q_NNM": "Non-nuclear-maximum charge",
    "rho_min": "Minimum density",
    "rho_min_ratio": "Relative minimum density",
    "rho_int_mean": "Mean interstitial density",
    "T_eigenvalues_t1": "Gradient-direction eigenvalue 1",
    "T_eigenvalues_t2": "Gradient-direction eigenvalue 2",
    "T_eigenvalues_t3": "Gradient-direction eigenvalue 3",
    "charge_FA": "Gradient fractional anisotropy",
    "shannon_entropy": "Shannon entropy",
    "fisher_information": "Fisher information",
    "disequilibrium": "Disequilibrium",
    "LMC_complexity": "LMC complexity",
}

# site heterogeneity: a statistic over the atoms of a per-atom quantity X
_STATS = {
    "site_std": "Site spread of",
    "site_range": "Site range of",
    "site_max": "Site maximum of",
    "site_min": "Site minimum of",
    "within_element_var": "Within-element variance of",
    "between_element_var": "Between-element variance of",
}
for _x in ("m1", "f_bond", "zeta"):
    for _suffix, _text in _STATS.items():
        NAMES[f"{_x}_{_suffix}"] = f"{_text} {NAMES[_x][0].lower()}{NAMES[_x][1:]}"


def label(name: str) -> str:
    """The descriptive name of a descriptor; any other name (a composition feature, a
    descriptor added later) is returned unchanged."""
    return NAMES.get(name, name)
