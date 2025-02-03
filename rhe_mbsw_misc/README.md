# **mBSW Model Fitting for the Rheology Textbook** 
*Created: Jan 31, 2025*

This repository contains a Jupyter Notebook (`misc_scraping.ipynb`) for data processing, interpolation, and curve fitting. The notebook reads CSV files, processes numerical data, and performs continuous spectrum fitting using `lmfit`.

## Summary

** An mBSW model is fitted to PB-Linear SAOS data (Fig. 1) from Hatzikiriakos et al. (2000)
** Specifically matched the glassy modulus, for pedagogical reasons. Achieved by varying the minimum relaxation time cutoff ($\tau_{\text{min}} = 1.8\times 10^{-5}$ s) 
** Generated the following figure and corresponding .csv files in the output_data/ folder
** Additional analysis performed for improved model fitting and uncertainty quantification

## Files

- .csv files are the scraped data from 
  > Hatzikiriakos, Savvas & Kapnistos, Michael & Chevillard, Cyril & Winter, H. & Roovers, Jacques. (2000). Relaxation time spectra of star polymers. Rheologica Acta. 39. 38-43. 10.1007/s003970050005.

- data_plot_functions.py – custom library for plot aesthetics
- wpd_project.tar - automeris.io file generated from the scraping

## Background

The exercise stems from the fact that the mBSW fit plot generated in the textbook for the above dataset was inaccurate in the glassy regime. The glassy modulus determined by the mBSW spectrum model is determined not by the model parameters but by the $\tau_{min}$ used in the numerical integration calculation for the dynamic moduli. One can demonstrate the effects by choosing different values for that variable in this code, for which the loop is provided. 


For practical purposes, the $\tau_{min}$ is chosen to be $5 \times 10^{-5}$ based on the visual judgment of the glassy modulus. The equation of the mBSW model chosen is:

$$H(\tau) = e^{-\left(\frac{\tau}{\tau_{\max}}\right)^\beta} 
\left[ H_e \left( \frac{\tau}{\tau_{\max}} \right)^{n_e} + 
H_g \left( \frac{\tau}{\tau_e} \right)^{-n_g} \right]$$


The experimental data extracted from the paper is the dynamic moduli of the “PB-linear”, a well-entangled linear 1,4 polybutadiene, 
<div align="center">
  <table>
    <tr>
      <td><img src="https://github.com/user-attachments/assets/1b0bb323-2c9e-4a0b-a92c-d03d693e7137" width="400"></td>
      <td><img src="https://github.com/user-attachments/assets/423a3ed2-a54b-44da-b95e-e236f7915a02" width="400"></td>
    </tr>
  </table>
</div>

The mBSW parameters proposed in the paper (transformed to the standard form above) are as follows:

<div align="center">

| **Variable**    | **Value**          |
|---------------|------------------|
| $H_g$ (Pa)   | $4.9 \times 10^5$ |
| $n_g$         | $0.67$            |
| $\tau_e$ (s)  | $0.86$            |
| $H_e$ (Pa)   | $1.86 \times 10^5$ |
| $n_e$         | $0.30$            |
| $\tau_{\max}$ (s) | $9.0 \times 10^5$ |
| $\beta$       | $2.0$ 

</div>

However, we noticed that the broader fit to the dynamic moduli data can be improved, and hence, a model fitting was performed using the lmfit package, which showed different values of the mBSW parameters than the ones published in the paper. This occurs as the authors have obtained these parameters by fitting the spectrum to the Prony series obtained from moduli instead of fitting directly to the data. Note that the fitting here is performed using the $w=S$ weighting in the residual definition, as there was no available estimation of uncertainties.

The updated parameters are given by,

<div align="center">

| **Variable**    | **Value**            |
|---------------|------------------|
| $H_g$ (Pa)   | $4.87 \times 10^5$ |
| $n_g$         | $0.90$            |
| $\tau_e$ (s)  | $0.0852$          |
| $H_e$ (Pa)   | $2.73 \times 10^5$ |
| $n_e$         | $0.2775$          |
| $\tau_{\max}$ (s) | $1.13 \times 10^6$ |
| $\beta$       | $5.0$             |

</div>

The fit results are as follows:

<p align="center">
  <img src="https://github.com/user-attachments/assets/8506cf36-7969-49cd-a4a3-2b31ea93f6ce" width="1000">
</p>

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


[def]: image.png