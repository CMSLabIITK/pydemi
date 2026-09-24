"""
pydemi.descriptors.registry
---------------------------
Metadata for every quantity pydemi computes: its entry number in the
Consolidated Descriptor Reference, family, kind, inputs and formula.

Kinds
-----
``descriptor``     a physical descriptor, as specified
``variant``        a recommended alternative for a flagged ambiguity, proposed by
                   the reference unless its note says otherwise (reported alongside,
                   never replacing, the specified form)
``cross_term``     combinatorial pairing with no stated physical derivation
                   (Tier 3 entries 33-37); report only if your own analysis
                   ranks it
``preprocessing``  regression transform, not a descriptor (entries 38-39)
``metadata``       pipeline/structure metadata, not a descriptor (29-30)
``field``          a per-voxel field, not a scalar (e.g. ELF_D, entry 72)
``site``           a per-atom array, not a scalar (e.g. site potentials, 74)
``dataset``        defined across a dataset, not per structure (82); see
                   :mod:`pydemi.descriptors.dataset`

Only the first five kinds are scalar per-structure values returned by
``compute_descriptors``.

Names follow the reference exactly. Where the reference's own formula
and its recommended fix disagree, the name keeps the specified formula
and the fix gets a new explicit name, so a column name never silently
changes meaning between old and new result files.
"""

from dataclasses import dataclass
from typing import Optional

DESCRIPTOR = "descriptor"
VARIANT = "variant"
CROSS_TERM = "cross_term"
PREPROCESSING = "preprocessing"
METADATA = "metadata"
FIELD = "field"
SITE = "site"
DATASET = "dataset"
SCALAR_KINDS = (DESCRIPTOR, VARIANT, CROSS_TERM, PREPROCESSING, METADATA)


@dataclass(frozen=True)
class DescriptorInfo:
    name: str
    entry: Optional[int]
    family: str
    kind: str
    inputs: str
    formula: str
    note: str = ""
    opt_in: bool = False      # not produced by compute_descriptors by default


def _elf_entries(p: str, inputs: str, note: str) -> list:
    """Family B entries 48-52 for ELF-like field prefix ``p`` (ELF or ELFD)."""
    return [
        DescriptorInfo(f"f_{p}_localized", 48, "B", DESCRIPTOR, f"{inputs} STRUCT",
                       "(1/N_bond) sum_bond 1(ELF > 0.5)", note),
        DescriptorInfo(f"{p}_bond_avg", 49, "B", DESCRIPTOR, f"{inputs} STRUCT",
                       "mean ELF over the bonding shell", note),
        DescriptorInfo(f"zeta_{p}", 50, "B", DESCRIPTOR, f"{inputs} STRUCT",
                       "1 - sum |grad ELF . r_hat| / sum |grad ELF| over r_k > c1",
                       note + "; core shell excluded"),
        DescriptorInfo(f"{p}_threshold_sweep_025", 51, "B", DESCRIPTOR, f"{inputs} STRUCT",
                       "f_ELF(t = 0.25) over the bonding shell", note),
        DescriptorInfo(f"{p}_threshold_sweep_075", 51, "B", DESCRIPTOR, f"{inputs} STRUCT",
                       "f_ELF(t = 0.75) over the bonding shell", note),
        DescriptorInfo(f"{p}_threshold_sweep_inflection", 51, "B", DESCRIPTOR,
                       f"{inputs} STRUCT", "t of steepest descent of f_ELF(t) (modal ELF)", note),
        DescriptorInfo(f"{p}_core_valence_contrast", 52, "B", DESCRIPTOR, f"{inputs} STRUCT",
                       "<ELF>_core / <ELF>_bond", note),
    ]


_SITE_ENTRY = {"m1": 53, "f_bond": 54, "zeta": 55}
_CELL_FORMULA = {
    "m1": "sum w rho r / sum w rho", "m2": "sum w rho r^2 / sum w rho",
    "sigma_r2": "m2 - m1^2", "f_core": "core-shell share of w rho",
    "f_bond": "bond-shell share of w rho", "f_int": "interstitial share of w rho",
    "zeta": "1 - sum w |grad rho . r_hat_i| / sum w |grad rho|",
}


def _site_entries(suffix: str = "", family: str = "C", note: str = "") -> list:
    """Family C statistics (53-59) for each site quantity."""
    out = []
    for X, entry in _SITE_ENTRY.items():
        out += [
            DescriptorInfo(f"{X}_site_std{suffix}", entry, family, DESCRIPTOR, "CHG STRUCT",
                           f"std over sites of {X}^(i)", note),
            DescriptorInfo(f"{X}_site_range{suffix}", 56, family, DESCRIPTOR, "DERIV",
                           f"max - min over sites of {X}^(i)", note),
            DescriptorInfo(f"{X}_site_max{suffix}", 57, family, DESCRIPTOR, "DERIV",
                           f"max over sites of {X}^(i)", note),
            DescriptorInfo(f"{X}_site_min{suffix}", 57, family, DESCRIPTOR, "DERIV",
                           f"min over sites of {X}^(i)", note),
            DescriptorInfo(f"{X}_var_within{suffix}", 58, family, DESCRIPTOR, "DERIV",
                           f"sum_e w_e Var_(i in e) {X}^(i), w_e = atom fraction",
                           (note + "; " if note else "") + "within-element: configurational disorder"),
            DescriptorInfo(f"{X}_var_between{suffix}", 59, family, DESCRIPTOR, "DERIV",
                           f"sum_e w_e (mean_e {X}^(i) - mean {X}^(i))^2",
                           (note + "; " if note else "") + "between-element: chemical differentiation"),
            DescriptorInfo(f"{X}_within_share{suffix}", 58, family, DESCRIPTOR, "DERIV",
                           "var_within / (var_within + var_between)",
                           (note + "; " if note else "") + "the single ANOVA ratio the reference recommends"),
        ]
    return out


def _partition_entries(scheme: str, entry: int, opt_in: bool = False) -> list:
    note = f"{scheme} partition (entry {entry}); compare with the nearest-atom value"
    if opt_in:
        note += "; opt-in, and the Becke product size k must be reported"
    cell = [DescriptorInfo(f"{X}_{scheme}", entry, "H", VARIANT, "CHG STRUCT",
                           _CELL_FORMULA[X], note, opt_in) for X in _CELL_FORMULA]
    sites = [DescriptorInfo(i.name, i.entry, "H", VARIANT, i.inputs, i.formula, i.note, opt_in)
             for i in _site_entries(f"_{scheme}", "H", note)]
    return cell + sites


_ENTRIES = [
    # ---------------- Tier 1 ----------------
    DescriptorInfo("zeta", 1, "tier1", DESCRIPTOR, "CHG STRUCT",
                   "1 - sum_k |grad rho_k . r_hat_k| / sum_k |grad rho_k|"),
    DescriptorInfo("m1", 2, "tier1", DESCRIPTOR, "CHG STRUCT",
                   "sum_k rho_k r_k / sum_k rho_k"),
    DescriptorInfo("m2", 3, "tier1", DESCRIPTOR, "CHG STRUCT",
                   "sum_k rho_k r_k^2 / sum_k rho_k"),
    DescriptorInfo("sigma_r2", 4, "tier1", DESCRIPTOR, "DERIV", "m2 - m1^2"),
    DescriptorInfo("f_core", 5, "tier1", DESCRIPTOR, "CHG STRUCT",
                   "sum_{r_k <= c1} rho_k / sum_k rho_k"),
    DescriptorInfo("f_bond", 6, "tier1", DESCRIPTOR, "CHG STRUCT",
                   "sum_{c1 < r_k <= c2} rho_k / sum_k rho_k"),
    DescriptorInfo("f_int", 7, "tier1", DESCRIPTOR, "CHG STRUCT",
                   "sum_{r_k > c2} rho_k / sum_k rho_k"),
    DescriptorInfo("lnf", 8, "tier1", DESCRIPTOR, "CHG",
                   "(1/N) sum_k 1(lap rho_k < 0)",
                   "voxel-count fraction; sensitive to the Laplacian numerics"),
    DescriptorInfo("lnf_rho", 8, "tier1", VARIANT, "CHG",
                   "sum_{lap rho < 0} rho_k / sum_k rho_k",
                   "charge-weighted lnf recommended by the reference"),
    DescriptorInfo("fint_over_lnf", 9, "tier1", DESCRIPTOR, "DERIV", "f_int / lnf"),
    DescriptorInfo("fint_over_lnf_rho", 9, "tier1", VARIANT, "DERIV", "f_int / lnf_rho"),
    DescriptorInfo("moment_ratio", 10, "tier1", DESCRIPTOR, "DERIV", "m2 / m1",
                   "carries units of length; not scale-invariant"),
    DescriptorInfo("moment_ratio_scale_free", 10, "tier1", VARIANT, "DERIV", "m2 / m1^2",
                   "the scale-invariant form the reference recommends"),
    DescriptorInfo("radial_cv", 11, "tier1", DESCRIPTOR, "DERIV", "sqrt(m2 - m1^2) / m1"),
    DescriptorInfo("zeta_over_rvar", 12, "tier1", DESCRIPTOR, "DERIV", "zeta / sigma_r2"),
    DescriptorInfo("zeta_over_sigma_r", 12, "tier1", VARIANT, "DERIV", "zeta / sigma_r",
                   "inverse length; scales consistently across cell sizes"),
    DescriptorInfo("charge_per_m1", 13, "tier1", DESCRIPTOR, "DERIV", "Q_tot / m1"),
    DescriptorInfo("lap_concentration", 14, "tier1", DESCRIPTOR, "CHG",
                   "sum_{lap<0} |lap rho_k| / sum_k |lap rho_k|",
                   "identically 1/2 for any periodic density: the cell integral of "
                   "a Laplacian vanishes, so negative and positive parts balance. "
                   "Carries no information; kept for completeness"),
    DescriptorInfo("lap_concentration_valence", 14, "tier1", VARIANT, "CHG STRUCT",
                   "same share over r_k > c1",
                   "pydemi proposal, not in the reference: excluding the core breaks "
                   "the identity (flux crosses the core boundary)"),
    DescriptorInfo("bond_int_ratio", 15, "tier1", DESCRIPTOR, "DERIV", "f_bond / f_int"),
    DescriptorInfo("Q_tot", None, "tier1", METADATA, "CHG", "sum_k rho_k dV",
                   "electron count; a sanity check, and the numerator of charge_per_m1"),

    # ---------------- Tier 2 (Magpie via matminer; no novelty claimed) ----------------
    DescriptorInfo("mean_mass", 16, "tier2", DESCRIPTOR, "COMP", "sum_i x_i A_i"),
    DescriptorInfo("max_mass", 17, "tier2", DESCRIPTOR, "COMP", "max_i A_i"),
    DescriptorInfo("mass_range", 18, "tier2", DESCRIPTOR, "COMP", "max_i A_i - min_i A_i"),
    DescriptorInfo("mean_elneg", 19, "tier2", DESCRIPTOR, "COMP", "sum_i x_i chi_i"),
    DescriptorInfo("elneg_diff", 20, "tier2", DESCRIPTOR, "COMP", "max_i chi_i - min_i chi_i"),
    DescriptorInfo("mean_vec", 21, "tier2", DESCRIPTOR, "COMP", "sum_i x_i VEC_i",
                   "Magpie NValence"),
    DescriptorInfo("max_vec", 22, "tier2", DESCRIPTOR, "COMP", "max_i VEC_i"),
    DescriptorInfo("mean_radius", 23, "tier2", DESCRIPTOR, "COMP", "sum_i x_i R_i",
                   "Magpie AtomicRadius, Angstrom"),
    DescriptorInfo("radius_diff", 24, "tier2", DESCRIPTOR, "COMP", "max_i R_i - min_i R_i"),
    DescriptorInfo("n_elements", 25, "tier2", DESCRIPTOR, "COMP", "count"),
    DescriptorInfo("ionicity", 26, "tier2", DESCRIPTOR, "COMP",
                   "1 - exp(-(elneg_diff)^2 / 4)", "Pauling; never looks at the density"),
    DescriptorInfo("mean_period", 27, "tier2", DESCRIPTOR, "COMP", "sum_i x_i P_i"),
    DescriptorInfo("crystal_system_int", 28, "tier2", DESCRIPTOR, "STRUCT",
                   "spglib crystal system, 1 = triclinic ... 7 = cubic"),
    DescriptorInfo("space_group_number", None, "tier2", METADATA, "STRUCT", "spglib"),
    DescriptorInfo("is_f_block", 29, "tier2", METADATA, "COMP",
                   "every element is f-block", "pipeline metadata, not a descriptor"),
    DescriptorInfo("has_f_block", 30, "tier2", METADATA, "COMP",
                   "any element is f-block", "pipeline metadata, not a descriptor"),
    # entry 31 ("split") is train/validation/test assignment and is not computed

    # ---------------- Tier 3 ----------------
    DescriptorInfo("laplacian_std", 32, "tier3", DESCRIPTOR, "CHG",
                   "std of lap rho over all k",
                   "dominated by near-nucleus spikes rather than bonding"),
    DescriptorInfo("laplacian_std_valence", 32, "tier3", VARIANT, "CHG STRUCT",
                   "std of lap rho over r_k > c1",
                   "core-excluded form the reference recommends"),
    DescriptorInfo("vec_x_lnf", 33, "tier3", CROSS_TERM, "DERIV", "mean_vec * lnf"),
    DescriptorInfo("elneg_x_lnf", 34, "tier3", CROSS_TERM, "DERIV", "mean_elneg * lnf"),
    DescriptorInfo("vec_over_rvar", 35, "tier3", CROSS_TERM, "DERIV", "mean_vec / sigma_r2"),
    DescriptorInfo("bond_over_lnf", 36, "tier3", CROSS_TERM, "DERIV", "f_bond / lnf"),
    DescriptorInfo("lnf_x_m1", 37, "tier3", CROSS_TERM, "DERIV", "lnf * m1"),
    DescriptorInfo("sqrt_zeta", 38, "tier3", PREPROCESSING, "DERIV", "sqrt(zeta)"),
    DescriptorInfo("log_lnf", 39, "tier3", PREPROCESSING, "DERIV", "ln(lnf)"),
] + _elf_entries("ELF", "ELF", "true ELF from ELFCAR; present only when one is loaded") \
  + _elf_entries("ELFD", "CHG", "on ELF_D reconstructed from rho (F1)") + [
    # ---------------- F1-F6, I1: fields derivable from rho alone ----------------
    DescriptorInfo("elf_d", 72, "F1", FIELD, "CHG",
                   "[1 + (D_P / C_F rho^{5/3})^2]^{-1}, Kirzhnits t_P (Tsirelson-Stash)",
                   "registered on the engine as field 'elf_d'"),
    DescriptorInfo("VH_site", 74, "F2", SITE, "CHG",
                   "V_H(R_i), Fourier-interpolated", "pydemi.descriptors.potential.site_potentials"),
    DescriptorInfo("VH_spread", 75, "F2", DESCRIPTOR, "CHG", "std over i of V_H(R_i), eV",
                   "electronic Hartree potential only (no ionic term)"),
    DescriptorInfo("VH_int_min", 76, "F2", DESCRIPTOR, "CHG STRUCT",
                   "min over r_k > c2 of V_H, eV", "relative to the cell-average potential"),
    DescriptorInfo("V_site", 74, "F2", SITE, "POT", "V(R_i) from LOCPOT"),
    DescriptorInfo("V_spread", 75, "F2", DESCRIPTOR, "POT", "std over i of V(R_i), eV",
                   "LOCPOT; present only when one is loaded"),
    DescriptorInfo("V_int_min", 76, "F2", DESCRIPTOR, "POT STRUCT",
                   "min over r_k > c2 of V, eV", "LOCPOT; relative to the cell average"),
    DescriptorInfo("f_NCI", 77, "F3", DESCRIPTOR, "CHG",
                   "(1/N) sum 1(s < 0.5 and rho < 0.05 a.u.)"),
    DescriptorInfo("NCI_attractive", 78, "F3", DESCRIPTOR, "CHG",
                   "fraction of NCI voxels with lambda2 < 0"),
    DescriptorInfo("sign_lambda2_rho_mean", 79, "F3", DESCRIPTOR, "CHG",
                   "mean of sign(lambda2) rho over NCI voxels, e/bohr^3"),
    DescriptorInfo("ellip_bond_avg", 80, "F4", DESCRIPTOR, "CHG STRUCT",
                   "<lambda1/lambda2 - 1> over bond voxels with lambda2 < 0",
                   "can be dominated by voxels with lambda2 -> 0^-"),
    DescriptorInfo("ellip_bond_median", 80, "F4", VARIANT, "CHG STRUCT",
                   "median of the same set", "pydemi addition: robust companion to the mean"),
    DescriptorInfo("ellip_bond_std", 81, "F4", DESCRIPTOR, "CHG STRUCT", "std of the same set"),
    DescriptorInfo("zeta_ellip_agreement", 82, "F4", DATASET, "DERIV",
                   "correlation of zeta with ellip_bond_avg across the dataset",
                   "pydemi.descriptors.dataset.zeta_ellip_agreement"),
    DescriptorInfo("f_H_negative", 83, "F5", DESCRIPTOR, "CHG STRUCT",
                   "(1/N_bond) sum_bond 1(H < 0), H = lap/4 - g (a.u.)"),
    DescriptorInfo("H_bond_mean", 84, "F5", DESCRIPTOR, "CHG STRUCT",
                   "<H> over the bonding shell, hartree/bohr^3"),
    DescriptorInfo("G_over_rho", 85, "F5", DESCRIPTOR, "CHG STRUCT",
                   "<g/rho> over the bonding shell, hartree/electron"),
    DescriptorInfo("shannon_entropy", 86, "F6", DESCRIPTOR, "CHG",
                   "-int rho~ ln rho~ dV, rho~ = rho/N_e, bohr units"),
    DescriptorInfo("fisher_information", 87, "F6", DESCRIPTOR, "CHG",
                   "int |grad rho~|^2 / rho~ dV, 1/bohr^2"),
    DescriptorInfo("disequilibrium", 88, "F6", DESCRIPTOR, "CHG", "int rho~^2 dV, 1/bohr^3"),
    DescriptorInfo("LMC_complexity", 89, "F6", DESCRIPTOR, "DERIV", "D e^S (unit-free)"),
    DescriptorInfo("T_eig_1", 102, "I1", DESCRIPTOR, "CHG",
                   "largest eigenvalue of T_ab = sum d_a rho d_b rho / sum |grad rho|^2"),
    DescriptorInfo("T_eig_2", 102, "I1", DESCRIPTOR, "CHG", "middle eigenvalue of T"),
    DescriptorInfo("T_eig_3", 102, "I1", DESCRIPTOR, "CHG", "smallest eigenvalue of T"),
    DescriptorInfo("charge_FA", 103, "I1", DESCRIPTOR, "DERIV",
                   "sqrt(3/2) ||T - I/3||_F / ||T||_F"),
] + _site_entries() + [
    DescriptorInfo("n_atoms", None, "C", METADATA, "STRUCT", "atoms in the cell",
                   "report with every site statistic: small cells give std over few sites"),
    # ---------------- Family E: spin density ----------------
    DescriptorInfo("M_abs", 63, "E", DESCRIPTOR, "SPIN", "sum |m| dV, mu_B"),
    DescriptorInfo("M_net", 64, "E", DESCRIPTOR, "SPIN", "|sum m dV|, mu_B",
                   "vector sum for non-collinear runs"),
    DescriptorInfo("m1_spin", 65, "E", DESCRIPTOR, "SPIN STRUCT", "sum |m| r / sum |m|",
                   "0 when not magnetic"),
    DescriptorInfo("sigma_r2_spin", 66, "E", DESCRIPTOR, "SPIN STRUCT",
                   "sum |m| r^2 / sum |m| - m1_spin^2", "0 when not magnetic"),
    DescriptorInfo("f_bond_spin", 67, "E", DESCRIPTOR, "SPIN STRUCT",
                   "sum_bond |m| / sum |m|", "0 when not magnetic"),
    DescriptorInfo("mu_site", 68, "E", SITE, "SPIN STRUCT", "sum_(k in i) m dV",
                   "pydemi.descriptors.spin.site_moments"),
    DescriptorInfo("mu_site_std", 69, "E", DESCRIPTOR, "SPIN STRUCT",
                   "sqrt(mean_i |mu_i - mean mu|^2)", "signed std when collinear"),
    DescriptorInfo("spin_frustration", 70, "E", DESCRIPTOR, "DERIV",
                   "1 - |sum mu_i| / sum |mu_i|", "vector norms; 0 when not magnetic"),
    DescriptorInfo("spin_charge_correlation", 71, "E", DESCRIPTOR, "CHG SPIN",
                   "Pearson r(rho_k, |m_k|)", "0 when not magnetic"),
    DescriptorInfo("is_spin_polarized", None, "E", METADATA, "SPIN",
                   "1 if the CHGCAR has magnetization blocks",
                   "every Family E value is the sentinel 0 when this is 0"),
    DescriptorInfo("is_magnetic", None, "E", METADATA, "SPIN",
                   "1 if M_abs > 0.01 mu_B per atom"),
    # ---------------- Family H: partition schemes ----------------
    DescriptorInfo("site_charge", None, "H", SITE, "CHG STRUCT",
                   "electrons per atom under a scheme", "pydemi.descriptors.sites.site_charges"),
] + _partition_entries("power", 99) + _partition_entries("becke", 100, opt_in=True) \
  + _partition_entries("hirshfeld", 101) + [
    DescriptorInfo("hirshfeld_charge", 101, "H", SITE, "CHG STRUCT REF",
                   "q_i = N_i - int w_i rho dV", "pydemi.descriptors.sites.hirshfeld_charges"),
    # ---------------- Family A: deformation density ----------------
    DescriptorInfo("m1_def", 40, "A", DESCRIPTOR, "AE REF", "sum |drho| r / sum |drho|",
                   "AECCAR route; NaN without AECCARs"),
    DescriptorInfo("m2_def", 41, "A", DESCRIPTOR, "AE REF", "sum |drho| r^2 / sum |drho|"),
    DescriptorInfo("sigma_r2_def", 42, "A", DESCRIPTOR, "DERIV", "m2_def - m1_def^2"),
    DescriptorInfo("f_bond_def", 43, "A", DESCRIPTOR, "AE REF STRUCT",
                   "sum_(bond, drho>0) drho / sum_(drho>0) drho"),
    DescriptorInfo("f_int_def", 44, "A", DESCRIPTOR, "AE REF STRUCT",
                   "sum_(int, drho>0) drho / sum_(drho>0) drho"),
    DescriptorInfo("f_bond_dep", 45, "A", DESCRIPTOR, "AE REF STRUCT",
                   "sum_(bond, drho<0) |drho| / sum_(drho<0) |drho|"),
    DescriptorInfo("bond_charge_transfer", 46, "A", DESCRIPTOR, "AE REF STRUCT",
                   "int_bond drho dV, electrons"),
    DescriptorInfo("def_polarity", 47, "A", DESCRIPTOR, "AE REF", "int |drho| dV / Q_tot"),
    DescriptorInfo("def_charge_mismatch", None, "A", METADATA, "AE REF",
                   "int drho dV, electrons", "should be ~0; large means wrong reference counts"),
    DescriptorInfo("def_all_electron", None, "A", METADATA, "AE",
                   "1 if AECCAR0 + AECCAR2 was the field"),
    # ---------------- Family D and entry 106: calibrated quantities ----------------
    DescriptorInfo("grid_ionicity", 60, "D", DESCRIPTOR, "CHG LIT",
                   "sigmoid(b . z(features)), fitted to Phillips f_i",
                   "NaN without an IonicityCalibration; extrapolation outside tetrahedral compounds"),
    DescriptorInfo("cohen_B0_predicted", 61, "D", DESCRIPTOR, "DERIV STRUCT",
                   "(1971 - 220 lambda) d^-3.5 GPa, lambda = Cohen's class 0/1/2",
                   "a baseline, not a training feature; formula extrapolation when cohen_in_scope = 0"),
    DescriptorInfo("cohen_in_scope", None, "D", METADATA, "COMP",
                   "1 for group-IV, III-V and II-VI (1:1) compounds"),
    DescriptorInfo("ionicity_residual", 62, "D", DESCRIPTOR, "DERIV",
                   "grid_ionicity - ionicity (Pauling)"),
    DescriptorInfo("B0_rho_proxy", 106, "D", DESCRIPTOR, "DERIV LIT",
                   "a (rho_mid_mean / bond_length_mean^3)^b, fitted to known B0",
                   "NaN without a BulkModulusCalibration"),
    # ---------------- entry 73: ELF_D validation ----------------
    DescriptorInfo("ELFD_fidelity_r", 73, "B", DESCRIPTOR, "CHG ELF",
                   "Pearson r(ELF_D, ELF) over ELFCAR voxels", "present only with an ELFCAR"),
    DescriptorInfo("ELFD_fidelity_mae", 73, "B", DESCRIPTOR, "CHG ELF", "mean |ELF_D - ELF|"),
    DescriptorInfo("ELFD_fidelity_r_bond", 73, "B", VARIANT, "CHG ELF STRUCT",
                   "Pearson r over the bonding shell", "where the comparison is meaningful"),
    DescriptorInfo("ELFD_fidelity_mae_bond", 73, "B", VARIANT, "CHG ELF STRUCT",
                   "mean |ELF_D - ELF| over the bonding shell"),
    # ---------------- Family G: topology and connectivity ----------------
    DescriptorInfo("rho_perc_a", 90, "G", DESCRIPTOR, "CHG",
                   "sup{c : {rho > c} has a cluster wrapping along a}, e/A^3",
                   "the reference's min{...} read as the supremum it describes"),
    DescriptorInfo("rho_perc_b", 90, "G", DESCRIPTOR, "CHG", "same along b"),
    DescriptorInfo("rho_perc_c", 90, "G", DESCRIPTOR, "CHG", "same along c"),
    DescriptorInfo("perc_anisotropy", 91, "G", DESCRIPTOR, "DERIV",
                   "(max - min) / mean of rho_perc_a/b/c"),
    DescriptorInfo("n_max", 92, "G", DESCRIPTOR, "CHG",
                   "PL maxima (Freudenthal 14-neighbour link)",
                   "all census counts are sensitive to ripple in near-flat low-density regions"),
    DescriptorInfo("n_min", 92, "G", DESCRIPTOR, "CHG", "PL minima"),
    DescriptorInfo("n_saddle1", 92, "G", DESCRIPTOR, "CHG",
                   "index-1 saddles, sum of (components(lower link) - 1)"),
    DescriptorInfo("n_saddle2", 92, "G", DESCRIPTOR, "CHG",
                   "index-2 saddles, sum of (components(upper link) - 1)"),
    DescriptorInfo("euler_consistency", 93, "G", METADATA, "DERIV",
                   "n_max - n_saddle2 + n_saddle1 - n_min",
                   "identically 0 for a consistent PL census (Banchoff): an implementation "
                   "self-check, not a grid-adequacy flag"),
    DescriptorInfo("n_NNM", 94, "G", DESCRIPTOR, "CHG STRUCT",
                   "maxima farther than max(c1, R_PAW) from their nearest nucleus",
                   "R_PAW = RCORE from POTCAR/OUTCAR, else covalent radius; counts every "
                   "ripple maximum, see n_NNM_significant"),
    DescriptorInfo("n_NNM_significant", 94, "G", VARIANT, "CHG STRUCT",
                   "non-nuclear maxima whose basin holds >= 0.01 e",
                   "pydemi addition: robust to low-amplitude ripple"),
    DescriptorInfo("n_NNM_persistent", 94, "G", VARIANT, "CHG STRUCT",
                   "non-nuclear maxima with relative persistence (peak - merge) / peak >= 0.1",
                   "pydemi addition: removes ripple, including flat free-electron seas"),
    DescriptorInfo("Q_NNM", 95, "G", DESCRIPTOR, "CHG STRUCT",
                   "charge in the steepest-ascent basins of the non-nuclear maxima"),
    DescriptorInfo("Q_NNM_persistent", 95, "G", VARIANT, "CHG STRUCT",
                   "charge of the persistent non-nuclear maxima, ripple basins merged in",
                   "pydemi addition"),
    DescriptorInfo("paw_radii_known", None, "G", METADATA, "POTCAR/OUTCAR",
                   "1 if the non-nuclear-maximum cutoffs used PAW RCORE values"),
    DescriptorInfo("rho_min", 96, "G", DESCRIPTOR, "CHG", "min_k rho_k, e/A^3",
                   "can be negative for PAW pseudo-densities"),
    DescriptorInfo("rho_min_int", 96, "G", VARIANT, "CHG STRUCT",
                   "min rho over r_k > max(c2, R_PAW), e/A^3",
                   "pydemi addition: the interstitial floor, outside every PAW sphere; "
                   "NaN when that region is empty"),
    DescriptorInfo("rho_min_ratio", 97, "G", DESCRIPTOR, "DERIV", "rho_min / <rho>_V",
                   "on PAW CHGCARs usually set by negative pseudo-density near a nucleus"),
    DescriptorInfo("rho_min_int_ratio", 97, "G", VARIANT, "DERIV", "rho_min_int / <rho>_V",
                   "pydemi addition: the metallicity criterion the reference intends"),
    DescriptorInfo("rho_int_mean", 98, "G", DESCRIPTOR, "CHG STRUCT",
                   "<rho_k> over r_k > c2 (volume mean), e/A^3"),
    # ---------------- I2: bond-strength descriptors ----------------
    DescriptorInfo("rho_mid_mean", 104, "I2", DESCRIPTOR, "CHG STRUCT",
                   "<rho(midpoint)> over first-shell bonds, e/A^3",
                   "first shell: d <= 1.1 d_i per atom"),
    DescriptorInfo("rho_mid_std", 105, "I2", DESCRIPTOR, "CHG STRUCT",
                   "std of rho(midpoint) over the bonds"),
    DescriptorInfo("n_bonds", None, "I2", METADATA, "STRUCT", "first-shell bond count"),
    DescriptorInfo("bond_length_mean", None, "I2", METADATA, "STRUCT",
                   "<d> over first-shell bonds, Angstrom",
                   "the explicit nearest-neighbour convention for the Cohen / rho-based B0"),
]

REGISTRY = {info.name: info for info in _ENTRIES}


def describe(name: str) -> DescriptorInfo:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown descriptor {name!r}") from None


def names(family: Optional[str] = None, kinds: Optional[tuple] = SCALAR_KINDS,
          opt_in: bool = False) -> list:
    """Registered names in reference order, filtered by family and/or kind.

    Defaults to the scalar kinds without opt-in entries; pass ``kinds=None``
    and ``opt_in=True`` for every entry.
    """
    return [i.name for i in _ENTRIES
            if (family is None or i.family == family)
            and (kinds is None or i.kind in kinds)
            and (opt_in or not i.opt_in)]
