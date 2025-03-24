This repository implements the framework for constructing diversified portfolios that simultaneously prevent risk concentrations at both the asset and factor levels, as proposed in the paper [*Asset and Factor Risk Budgeting: a balanced approach*](https://www.tandfonline.com/doi/full/10.1080/14697688.2024.2435627).

## Notebooks Overview

- **01_asset_factor_rb_intro**: Illustrates standard risk-based portfolio construction methods and introduces the Asset-Factor Risk Budgeting framework under a simple risk model.

- **02_barra_equity_model_application**: Applies the proposed approach within a real-world equity risk model (Barra) to analyze the nature of the resulting portfolios.

- **03_expected_shortfall_extension_under_macro_factor_model**: Demonstrates how to compute such portfolios when risk is quantified by Expected Shortfall instead of volatility, using a macro factor model for a multi-asset portfolio.

- **04_choice_of_importance_parameters**: Discusses how to balance asset-level and factor-level risk contributions through the choice of importance parameters.



