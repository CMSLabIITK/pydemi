"""The data files of the paper (available from the corresponding author on request) -> data/deposit/.

Writes the descriptors of the 6,059 DFT densities and of the 605 ChargE3Net test densities, the
property targets, the test-set predictions of LightGBM (17 properties and the two columns of
Kumar et al. used in Table 4), of the models on DFT and predicted descriptors (Fig. 5) and of
MatTen (Table 4), and the PAW dataset of every calculation, read from its OUTCAR. Every table
is checked against the results it is copied from; their README is
deposit_README.md next to this script.
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score

import pydemi

HERE = Path(__file__).resolve().parent
PP = HERE.parent
DATA = PP / "data"
OUT = DATA / "deposit"
DATASET = Path("/data/sai/new_charge/6000_data_aug13")
MATTEN = Path.home() / "secondPaper/MatTen"
NAMES = list(pydemi.descriptor_names())
ID = ["structure", "material_id", "formula", "split"]
JCIM = {"energy_jcim": "formation_energy_per_atom", "debye_jcim": "debye_temperature"}


def descriptors() -> pd.DataFrame:
    d = pd.read_csv(DATA / "descriptors.csv")
    assert list(d.columns) == ID + NAMES and len(d) == 6059 and d[NAMES].notna().all().all()
    d.to_csv(OUT / "descriptors_dft.csv", index=False)
    ml = pd.read_csv(DATA / "descriptors_chargE3net_test605.csv")
    assert list(ml.columns) == ["structure"] + NAMES and len(ml) == 605
    ml = d[ID].merge(ml, on="structure", how="right", validate="one_to_one")
    assert ml.material_id.notna().all()
    ml.to_csv(OUT / "descriptors_chargE3net_test605.csv", index=False)
    cat = pd.read_csv(DATA / "descriptors_columns.csv").drop(columns="references")
    assert list(cat.name) == NAMES
    cat.to_csv(OUT / "descriptors_columns.csv", index=False)
    return d


def targets(d: pd.DataFrame) -> pd.DataFrame:
    t = pd.read_csv(DATA / "targets.csv")
    assert t.structure.tolist() == d.structure.tolist()
    p = pd.read_csv(DATA / "properties.csv").set_index("formula_sp")
    for k, col in JCIM.items():
        t[k] = p.loc[t.structure, col].to_numpy()
    t.to_csv(OUT / "targets.csv", index=False)
    cols = pd.read_csv(DATA / "targets_columns.csv")
    extra = [{"column": "energy_jcim", "units": "eV/atom", "source": "properties.csv (formation_energy_per_atom)",
              "task": "regression", "description": "energy relative to the free atoms (E_atoms of Table 4)"},
             {"column": "debye_jcim", "units": "K", "source": "properties.csv (debye_temperature)",
              "task": "regression", "description": "Debye temperature of Kumar et al. (Theta_D of Table 4)"}]
    for e in extra:
        y = t[e["column"]]
        e.update(n=int(y.notna().sum()), n_train=int(y[t.split == "train"].notna().sum()),
                 n_test=int(y[t.split == "test"].notna().sum()))
    pd.concat([cols, pd.DataFrame(extra)]).to_csv(OUT / "targets_columns.csv", index=False)
    return t


def lightgbm(d: pd.DataFrame, t: pd.DataFrame) -> None:
    mid = d.set_index("material_id")["structure"]
    rows = []
    for f in sorted((PP / "results/study_v2/jobs").glob("*__desc.json")):
        job = json.loads(f.read_text())
        m = job["models"]["lgbm"]
        y, p = np.array(job["test_y"]), np.array(m["ensemble"]["test_p"])
        cls = job["task"] == "classification"
        score = roc_auc_score(y, p) if cls else r2_score(y, p)
        assert np.isclose(score, m["ensemble"]["roc_auc" if cls else "r2"]), f.name
        s = mid.loc[job["test_material_id"]].to_numpy()
        assert np.allclose(t.set_index("structure").loc[s, job["target"]], y, atol=1e-9), f.name
        part = pd.DataFrame({"structure": s, "material_id": job["test_material_id"], "target": job["target"],
                             "true": y, "pred_mean": p})
        for r in m["seeds"]:
            part[f"pred_seed{r['seed']}"] = r["test_p"]
        rows.append(part)
    pd.concat(rows).to_csv(OUT / "predictions_lightgbm_test.csv", index=False)


def dft_vs_ml() -> None:
    p = pd.read_csv(PP / "results/study_v2_mlheldout/predictions.csv")
    s = pd.read_csv(PP / "results/study_v2_mlheldout/summary.csv").set_index("target")
    for k, g in p.groupby("target"):
        if np.isfinite(s.loc[k, "dft_r2"]):
            assert np.isclose(r2_score(g.true, g.dft_desc), s.loc[k, "dft_r2"]), k
            assert np.isclose(r2_score(g.true, g.ml_desc), s.loc[k, "ml_r2"]), k
    p = p.rename(columns={"dft_desc": "pred_dft_descriptors", "ml_desc": "pred_ml_descriptors"})
    p.to_csv(OUT / "predictions_dft_vs_ml_test605.csv", index=False)


def matten(t: pd.DataFrame) -> None:
    files = {"bulk_modulus": ("bulk", "bulk_hill"), "shear_modulus": ("shear", "shear_hill"),
             "youngs_modulus": ("youngs", "youngs_modulus"),
             "energy_jcim": ("formation", "formation_energy_per_atom"),
             "debye_jcim": ("debye", "debye_temperature")}
    tt = t.set_index("structure")
    rows = []
    for k, (f, col) in files.items():
        m = pd.read_csv(MATTEN / f"baseline_test_predictions_{f}.csv")
        assert (tt.loc[m.formula_sp, "split"] == "test").all()
        assert np.allclose(m[f"{col}_true"], tt.loc[m.formula_sp, k], atol=1e-3), k
        rows.append(pd.DataFrame({"structure": m.formula_sp, "target": k, "true": tt.loc[m.formula_sp, k].to_numpy(),
                                  "pred": m[f"{col}_pred"]}))
    pd.concat(rows).to_csv(OUT / "predictions_matten_test.csv", index=False)


def paw_one(d: Path) -> dict:
    f = d / "OUTCAR"
    if not f.exists():
        return {"structure": d.name, "outcar": False, "elements": "", "paw_datasets": "", "zval": ""}
    tit, zv = [], []
    with open(f, errors="ignore") as fh:
        for line in fh:
            if "TITEL" in line:
                tit.append(line.split("=", 1)[1].strip())
            elif "ZVAL" in line and "POMASS" in line:
                zv.append(line.split("ZVAL")[1].split("=")[1].split()[0])
            elif "Iteration" in line:
                break
    with open(d / "CHGCAR") as fh:
        el = [next(fh) for _ in range(6)][5].split()
    assert len(el) == len(tit) == len(zv), d.name
    return {"structure": d.name, "outcar": True, "elements": ";".join(el), "paw_datasets": ";".join(tit),
            "zval": ";".join(zv)}


def paw(d: pd.DataFrame) -> None:
    with ProcessPoolExecutor(16) as ex:
        rows = list(ex.map(paw_one, [DATASET / s for s in d.structure], chunksize=50))
    p = pd.DataFrame(rows)
    p.to_csv(OUT / "paw_datasets.csv", index=False)
    per = {}
    for r in p[p.outcar].itertuples():
        for e, ti, z in zip(r.elements.split(";"), r.paw_datasets.split(";"), r.zval.split(";")):
            per.setdefault(e, set()).add((ti, float(z)))
    assert all(len(v) == 1 for v in per.values()), {e: v for e, v in per.items() if len(v) > 1}
    pd.DataFrame([{"element": e, "paw_dataset": next(iter(v))[0], "zval": next(iter(v))[1]}
                  for e, v in sorted(per.items())]).to_csv(OUT / "paw_by_element.csv", index=False)
    print(f"PAW: {int(p.outcar.sum())} OUTCARs, {len(per)} elements, one dataset per element")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    d = descriptors()
    t = targets(d)
    lightgbm(d, t)
    dft_vs_ml()
    matten(t)
    paw(d)
    shutil.copy(HERE / "deposit_README.md", OUT / "README.md")
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:40s} {f.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
