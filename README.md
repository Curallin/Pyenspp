# Pyenspp

A Python package for ensemble precipitation forecast post processing.

* Created by **[Fuxuan Jiang](https://github.com/Curallin)**
* Free software: MIT License

## Features

* `Pyenspp` is a Python package for post-processing of ensemble precipitation forecast.
* The functionality of the `pyenspp` is organized into four main components: preprocessing, forecast calibration, dependence reconstruction, and forecast verification. 
* The package was evaluated using the ECMWF Sub-seasonal to Seasonal (S2S) dataset over the North River catchment. The case study demonstrates that the KAN–CSGD model implemented in `Pyenspp` significantly improves forecast reliability and accuracy.
* Example data and scripts are provided in the `examples/` directory to demonstrate the basic usage of `Pyenspp`.
* Designed with modularity in mind, `Pyenspp` provides a flexible foundation for future enhancements, and we welcome community-driven improvements.

## Development

To set up for local development:

```bash
# Clone your fork
git clone git@github.com:Curallin/pyenspp.git
cd pyenspp

# Install in editable mode with live updates
pip install -e .
```

This installs the CLI globally but with live updates - any changes you make to the source code are immediately available when you run `pyenspp`.

## Author

`Pyenspp` was created in 2026 by Fuxuan Jiang.
