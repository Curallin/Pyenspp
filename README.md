# Pyenspp

A Python package for ensemble precipitation forecast post processing.

* Created by **[Fuxuan Jiang](https://github.com/Curallin)**
* Free software: MIT License

## Features

* `Pyenspp` is a Python package for post-processing of ensemble precipitation forecast.
* The functionality of the `pyenspp` is organized into four main components: preprocessing, forecast calibration, dependence reconstruction, and forecast verification. 
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

## Current Version

`Pyenspp` is currently in its first public release (**v0.1.0**), which provides the core framework for ensemble precipitation forecast post-processing, including preprocessing, calibration, dependence reconstruction, and verification.

As the initial release, the package focuses on establishing a modular and extensible foundation for future development.

## Roadmap

Future versions of `Pyenspp` are expected to include:

* Additional post-processing models such as EMOS, meta-Gaussian model (MGB) and distributional regression networks (DRNs).
* Enhanced visualization and diagnostic tools for forecast assessment.

## Author

`Pyenspp` was created in 2026 by Fuxuan Jiang.
