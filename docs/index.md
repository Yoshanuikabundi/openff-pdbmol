# OpenFF PDBScan

Stats collection from the PDB databank.

## Installing OpenFF PDBScan

OpenFF recommends using Conda virtual environments for all scientific Python work. PDBScan can be installed into a new Conda environment named `pdbscan` with the [`openff-pdbscan`] package:

```shell-session
$ mamba create -n pdbscan -c conda-forge openff-pdbscan
$ mamba activate pdbscan
```

If you do not have Conda or Mamba installed, see the [OpenFF installation documentation](inv:openff.docs#install).

[`openff-pdbscan`]: https://anaconda.org/conda-forge/openff-pdbscan

:::{toctree}
---
hidden: true
---

Overview <self>
:::

<!--
The autosummary directive renders to rST,
so we must use eval-rst here
-->
```{eval-rst}
.. raw:: html

    <div style="display: None">

.. autosummary::
   :recursive:
   :caption: API Reference
   :toctree: api/generated
   :nosignatures:

   openff.pdbscan

.. raw:: html

    </div>
```
