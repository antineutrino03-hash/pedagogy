# Miscellaneous Data Scraping and Analysis

This repository contains a Jupyter Notebook (`misc_scraping.ipynb`) for data processing, interpolation, and curve fitting. The notebook reads CSV files, processes numerical data, applies interpolation techniques, and performs curve fitting using `lmfit`.

## Files

- .csv files are the scraped data from – Hatzikiriakos, Savvas & Kapnistos, Michael & Chevillard, Cyril & Winter, H. & Roovers, Jacques. (2000). Relaxation time spectra of star polymers. Rheologica Acta. 39. 38-43. 10.1007/s003970050005.
- data_plot_functions.py – custom library for plot aesthetics
- wpd_project.tar - automeris.io file generated from the scraping

## Background

The exercise stems from the fact that the mBSW fit plot generated in the textbook for the above dataset was inaccurate in the glassy regime. The glassy modulus determined by the mBSW spectrum model is determined not by the model parameters, but the $\tau_{min}$ using in the numerical integration calculation for the dynamic moduli. One can demonstrate the effects by choosing different values for that variable in this code, for which the loop is provided. For practical purposes, the $\tau_{min}$ chosen to be $5 \times 10^{-5}$ based on visual judgement of the glassy modulus.


However, we noticed that the broader fit to the dynamic moduli data can be improved, and hence, a model fitting was performed using the lmfit package, which showed different values of the mBSW parameters than the ones published in the paper. This occurs as the authors have obtained these parameters by fitting the spectrum to the Prony series obtained from moduli instead of fitting directly to the data. Note that the fitting here is performed using the $w=S$ weighting in the residual definition, as there was no available estimation of uncertainties.

## Requirements

Ensure you have the following dependencies installed:

```bash
pip install pandas numpy matplotlib scipy lmfit
```

## Usage

- Load the notebook - misc_scraping.ipynb and write the file paths corresponding to your setup
- Store the excel files in the same location
- If you want to remove the model-fitting part, comment out the corresponding section. But you also need to modify the looping lists for params in the last cell

The notebook generates interpolated data points.
Curve fitting results are displayed with parameter optimizations.
Visualizations include raw data plots and fitted curves.

## Contributing

Feel free to modify the notebook for different datasets or additional processing steps. Please contact asm[eighteen][at]illinois.edu for any feedback.

## License

This project is open-source and licensed under the MIT License.
