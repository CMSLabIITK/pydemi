# Python API

Everything most users need is at the top level of `pydemi`. The optional modules
`pydemi.ml` and `pydemi.props` need the `ml` and `props` extras.

## Reading and writing

```{eval-rst}
.. autofunction:: pydemi.read_chgcar
.. autofunction:: pydemi.write_volumetric
.. autofunction:: pydemi.io.vasp.read_blocks
```

## Descriptors

```{eval-rst}
.. autofunction:: pydemi.featurize
.. autofunction:: pydemi.featurize_batch
.. autofunction:: pydemi.batch.find_runs
.. autofunction:: pydemi.descriptor_names
.. autofunction:: pydemi.catalogue
```

## Data structures

```{eval-rst}
.. autoclass:: pydemi.VolumetricData
   :members:
.. autoclass:: pydemi.Structure
   :members:
.. autoclass:: pydemi.Lattice
   :members:
.. autoclass:: pydemi.Grid
   :members:
.. autoclass:: pydemi.core.geometry.Shells
```

## Predicted densities: `pydemi.ml`

```{eval-rst}
.. autofunction:: pydemi.ml.predict
.. autofunction:: pydemi.ml.predict_grid
.. autofunction:: pydemi.ml.read_structure
.. autofunction:: pydemi.ml.vasp_grid_shape
.. autofunction:: pydemi.ml.load
.. autofunction:: pydemi.ml.n_valence
.. autofunction:: pydemi.ml.unsupported
.. autofunction:: pydemi.ml.missing
.. autoexception:: pydemi.ml.Cancelled
```

## Property estimates: `pydemi.props`

```{eval-rst}
.. autofunction:: pydemi.props.predict
.. autofunction:: pydemi.props.outside_range
.. autofunction:: pydemi.props.load
.. autofunction:: pydemi.props.model_path
.. autofunction:: pydemi.props.missing
.. autofunction:: pydemi.props.load_error
.. autoclass:: pydemi.props.Bundle
   :members: match, groups
.. autoclass:: pydemi.props.Target
   :members: score, error
.. autoclass:: pydemi.props.Match
```

## Validation: `pydemi.validate`

```{eval-rst}
.. autofunction:: pydemi.sensitivity_sweep
.. autofunction:: pydemi.validate.convergence.grid_convergence
.. autofunction:: pydemi.validate.convergence.analytic_convergence
.. autofunction:: pydemi.validate.convergence.recommended_spacing
.. automodule:: pydemi.validate.invariance
   :members: supercell, translate, rotate, compare
.. automodule:: pydemi.validate.analytic
   :members:
```

## Constants

```{eval-rst}
.. automodule:: pydemi.constants
   :members:
   :undoc-members:
```
