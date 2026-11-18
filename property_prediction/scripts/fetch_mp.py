"""Download Materials Project data for the 6,059 structures of properties.csv (read-only API use).

The API key is read by the MP client from ~/.config/.pmgrc.yaml (PMG_MAPI_KEY) or the
MP_API_KEY environment variable; it is never printed or written anywhere by this script.

Writes, into ../data:
  mp_summary.csv      one row per material: formation energy, band gap, metallicity, direct
                      gap and refractive index
  mp_elasticity.csv   one row per material with elastic data: VRH moduli, Debye temperature,
                      sound velocities, minimum thermal conductivities, the 21 independent
                      IEEE elastic constants C11..C66
  mp_cohesive.csv     the cohesive energy per atom
  mp_raw.json.gz      the documents as returned, for provenance
  mp_provenance.txt   date, database and client versions, counts, ids MP did not return
"""

import datetime as dt
import gzip
import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from emmet.core.mpid import AlphaID
from mp_api.client import MPRester

HERE = Path(__file__).resolve().parents[1] / "data"
IDS = pd.read_csv(HERE / "properties.csv")["material_id"].tolist()

SUMMARY_FIELDS = [
    "material_id", "formula_pretty", "formation_energy_per_atom", "band_gap", "is_metal",
    "is_gap_direct", "n",
]
ELASTIC_FIELDS = [
    "material_id", "elastic_tensor", "bulk_modulus", "shear_modulus", "debye_temperature",
    "sound_velocity", "thermal_conductivity",
]


def flat(doc: dict, prefix: str = "") -> dict:
    """Nested dicts as prefix_key columns; the 6x6 IEEE elastic tensor as C11..C66."""
    out = {}
    for k, v in doc.items():
        key = f"{prefix}{k}"
        if k == "elastic_tensor" and isinstance(v, dict):
            m = v.get("ieee_format")
            if m:
                for i in range(6):
                    for j in range(i, 6):
                        out[f"C{i + 1}{j + 1}"] = m[i][j]
        elif isinstance(v, dict):
            out.update(flat(v, f"{key}_"))
        elif isinstance(v, (list, tuple)):
            out[key] = json.dumps(v, default=str)
        else:
            out[key] = v
    return out


def legacy(material_id: str) -> str:
    """The 2026 database returns new ids (mp-aaaaaaft); properties.csv has the legacy ones (mp-149)."""
    return f"mp-{int(AlphaID(material_id))}"


def docs(results) -> list[dict]:
    out = [r.model_dump(mode="json") if hasattr(r, "model_dump") else dict(r) for r in results]
    for d in out:
        d["material_id_legacy"] = legacy(str(d["material_id"]))
    return out


def main() -> None:
    started = dt.datetime.now().isoformat(timespec="seconds")
    with MPRester() as mpr:
        db = mpr.get_database_version()
        summary = docs(mpr.materials.summary.search(material_ids=IDS, fields=SUMMARY_FIELDS))
        elastic = docs(mpr.materials.elasticity.search(material_ids=IDS, fields=ELASTIC_FIELDS))
        coh = mpr.get_cohesive_energy(IDS, normalization="atom")

    wanted = set(SUMMARY_FIELDS) | set(ELASTIC_FIELDS) | {"material_id_legacy"}
    keep = lambda d: {k: v for k, v in d.items() if k in wanted}
    pd.DataFrame([flat(keep(d)) for d in summary]).to_csv(HERE / "mp_summary.csv", index=False)
    pd.DataFrame([flat(keep(d)) for d in elastic]).to_csv(HERE / "mp_elasticity.csv", index=False)
    pd.DataFrame({"material_id_legacy": list(coh), "cohesive_energy_per_atom": list(coh.values())}
                 ).to_csv(HERE / "mp_cohesive.csv", index=False)
    with gzip.open(HERE / "mp_raw.json.gz", "wt") as f:
        json.dump({"summary": summary, "elasticity": elastic}, f, default=str)

    got_s = {d["material_id_legacy"] for d in summary}
    got_e = {d["material_id_legacy"] for d in elastic}
    missing = sorted(set(IDS) - got_s)
    (HERE / "mp_provenance.txt").write_text(
        f"fetched: {started} to {dt.datetime.now().isoformat(timespec='seconds')}\n"
        f"MP database version: {db} (new-style ids; material_id_legacy matches properties.csv)\n"
        f"mp-api {version('mp-api')}, emmet-core {version('emmet-core')}, pymatgen {version('pymatgen')}\n"
        f"requested: {len(IDS)} material ids (properties.csv)\n"
        f"summary docs: {len(summary)}, elasticity docs: {len(elastic)}, cohesive energies: "
        f"{sum(v is not None for v in coh.values())}\n"
        f"ids without a summary doc ({len(missing)}): {' '.join(missing)}\n"
        f"ids without elasticity data: {len(set(IDS) - got_e)}\n")
    print((HERE / "mp_provenance.txt").read_text())


if __name__ == "__main__":
    main()
