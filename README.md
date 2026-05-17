# Pyenspp

A Python package for ensemble precipitation forecast post processing.

* Created by **[Fuxuan Jiang](https://github.com/Curallin)**
* Free software: MIT License
* Current Version: v0.2.0 (Updated May 2026)

## Features

* `Pyenspp` is a Python package for post-processing of ensemble precipitation forecast.
* Pyenspp implements several calibration methods, including Quantile Mapping (QM), KAN-CSGD, and Censored Shifted Gamma Ensemble Model Output Statistics (CSG-EMOS).
* The verification module provides more than 20 metrics including probabilistic (ensemble) metrics and deterministic metrics.
* The package was evaluated using the ECMWF Sub-seasonal to Seasonal (S2S) dataset over the North River catchment. The case study demonstrates that the KAN–CSGD model implemented in `Pyenspp` significantly improves forecast reliability and accuracy.
* Example data and scripts are provided in the `examples/` directory to demonstrate the basic usage of `Pyenspp`.
* Designed with modularity in mind, `Pyenspp` provides a flexible foundation for future enhancements, and we welcome community-driven improvements.

## Development

To set up for local development:

```bash
# Clone your fork
git clone https://github.com/Curallin/Pyenspp.git
cd pyenspp

# Install in editable mode with live updates
pip install -e .
```

This installs the CLI globally but with live updates - any changes you make to the source code are immediately available when you run `pyenspp`.

## Version

1. first public release (v0.1.0, Feb 2026)

`Pyenspp`  (**v0.1.0**) provides the core framework for ensemble precipitation forecast post-processing, including preprocessing, calibration, dependence reconstruction, and verification.

As the initial release, the package focuses on establishing a modular and extensible foundation for future development.

2. second public release (0.2.0, May 2026)

Added a CSG-EMOS calibration method with SCE-UA for parameter optimization, together with sample data and reproducible code examples in `examples/6.csgemos_usage.ipynb`.

## Roadmap

Future versions of `Pyenspp` are expected to include:

* Additional post-processing models such as meta-Gaussian model (MGB) and distributional regression networks (DRNs).
* Enhanced visualization and diagnostic tools for forecast assessment.

## Author

`Pyenspp` was created in 2026 by Fuxuan Jiang.
