# Command line

```text
pydemi featurize CHGCAR --out features.json
pydemi batch ./runs --glob "*/CHGCAR" --out features.csv --workers 8
pydemi catalogue --out catalogue.csv
pydemi sweep CHGCAR --param b --range 1.0:1.5:0.02 --out sweep.csv
pydemi gui
pydemi predict Si.cif --out-dir predicted --featurize
```

The options shared by `featurize`, `batch` and `sweep`:

| option | default | meaning |
|---|---|---|
| `--domains` | all | comma-separated: `bonding,structural,heterogeneity` |
| `--partition` | `nearest` | `nearest` or `becke` ({doc}`../guide/partitions`) |
| `--shells a,b` | `0.79,1.14` | sphere factors ({doc}`../guide/regions`) |
| `--laplacian-method` | `metric` | `metric` or `diagonal` ({doc}`../guide/numerics`) |
| `--derivative-backend` | `fft` | `fft` or `fd` |
| `--fd-order` | 4 | 2, 4, 6 or 8 with `fd` |
| `--float32` | off | single precision |

The full help of every command, as printed by `pydemi <command> --help`:

```{include} ../_generated/md/cli.md
```
