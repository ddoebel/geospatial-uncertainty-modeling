# Stochastic Weather Uncertainty Modeling

This project implements a physically motivated, stochastic uncertainty model for gridded weather data (ERA5). It leverages a modern data engineering pipeline to transform raw geospatial data into interpretable predictability metrics using EOF decomposition and Ornstein–Uhlenbeck (OU) processes.

## Project Overview

The pipeline quantifies how forecast uncertainty grows over time by modeling the temporal dynamics of reduced-order atmospheric states. By fitting continuous-time stochastic processes to historical anomalies, we derive analytical uncertainty growth curves and reconstruct them into spatial maps on the Earth's surface.

### Key Features
- **Data Engineering Pipeline**: Structured as composable assets compatible with Dagster, ensuring modularity and reproducibility.
- **Geospatial Analysis**: Uses `xarray` for labeled multidimensional data handling and `numpy`/`statsmodels` for numerical analysis.
- **Dimensionality Reduction**: EOF (Empirical Orthogonal Function) decomposition to capture dominant modes of variability.
- **Stochastic Modeling**: Ornstein–Uhlenbeck process fitting to quantify memory time scales and noise strength.
- **Uncertainty Mapping**: Closed-form reconstruction of lead-time-dependent spatial uncertainty.

## Pipeline Architecture

1.  **Ingestion & Validation**: Loads raw NetCDF data and performs structural (monotonicity, duplicates) and physical (range checks, units) validation.
2.  **Preprocessing**: Removes seasonal cycles (anomalies) and applies latitude-dependent cosine weighting to account for grid cell area differences.
3.  **EOF Decomposition**: Reduces spatial complexity by projecting anomalies onto a low-dimensional subspace of principal components.
4.  **OU Parameter Estimation**: Fits OU processes to the leading EOF coefficients. The model estimates:
    - **Decay Rate ($\lambda$)**: The inverse of the memory time scale.
    - **Noise Strength ($\sigma$)**: The magnitude of stochastic forcing.
5.  **Uncertainty Quantification**: Computes the variance growth $V(t) = \frac{\sigma^2}{2\lambda}(1 - e^{-2\lambda t})$ and projects it back to the original spatial grid.

## Core Methodology

The project bridges data engineering and physical climate science:
- **Physical Motivation**: Weather anomalies often exhibit "memory" that decays over time, a characteristic well-captured by the OU process (the continuous-time analog of AR(1)).
- **Mathematical Modeling**: Parameters are derived from the integral time scale of the autocorrelation function, ensuring consistent estimation of temporal persistence.
- **Interpretability**: The model provides metrics on where and how fast atmospheric predictability saturates.

## Roadmap

1.  **Improved OU Validation**: Add diagnostics for each EOF mode, including empirical vs. theoretical autocorrelation plots and automatic rejection of modes that violate OU assumptions.
2.  **Predictability Horizon Metric**: Calculate time-to-saturation metrics per mode to quantify the intuitive "memory" and usefulness of forecasts.
3.  **Spatial Uncertainty Snapshots**: Generate static maps for fixed lead times (1, 6, 12 months) to visualize the spatial progression of uncertainty.
4.  **Kalman Filter on Reduced State**: Implement state estimation by fusing noisy EOF coefficients with OU dynamics, mimicking data assimilation.
5.  **Code Refactoring**: Organize the codebase into dedicated modules (ingestion, preprocessing, EOF reduction, stochastic modeling, visualization) with improved documentation.

## Project Structure

The project is organized into a modular package structure:

- `src/stochastic_weather/core/`: Contains the pure mathematical and physical logic (EOF decomposition, OU process fitting, uncertainty calculations, and visualization).
- `src/stochastic_weather/assets/`: Defines the Dagster assets that orchestrate the data pipeline, now including rich metadata visualization in the Dagster UI.
- `data/`: Placeholder for input NetCDF weather data.
- `notebooks/`: Exploratory analysis and prototyping.

## Setup & Usage

### Prerequisites
- Python 3.9+
- Dependencies: `xarray`, `numpy`, `pandas`, `dagster`, `statsmodels`, `matplotlib`, `netcdf4`

### Data Preparation
To reproduce the results, you need the ERA5 Monthly Evaporation dataset.

1.  **Download the data**: [ERA5_LowRes_Monthly_evap.nc](https://cluster.klima.uni-bremen.de/~fmaussion/teaching/climate/ERA5_LowRes_Monthly_evap.nc)
2.  **Place the file**: Create a `data/` directory in the project root and place the downloaded file there.

The expected path is:
`data/ERA5_LowRes_Monthly_evap.nc`

### Running the Pipeline
The project is structured with Dagster assets. You can launch the Dagster UI to explore and run the pipeline:

```bash
dagster dev -f definitions.py
```

Or run it programmatically using the Dagster Python API.

---
*This is a student project developed to explore the intersection of data engineering and stochastic atmospheric dynamics.*
