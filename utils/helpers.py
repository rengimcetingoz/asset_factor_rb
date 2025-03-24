import numpy as np
import cvxpy as cp
import optuna
import statsmodels.api as sm
import pandas as pd

def is_pos_def(x):
    # check if a matrix is positive definite
    return np.all(np.linalg.eigvals(x) > 0)

def vol(x, cov):
    # compute the volatility of a portfolio
    return np.sqrt(x.T@cov@x)

def grad_vol(x, cov):
    # compute the gradient of the portfolio volatility
    return (cov@x)/vol(x, cov)

def gradient_expected_shortfall(sample, x, alpha):
    # compute the gradient of the expected shortfall
    return -np.mean(sample[sample@x<np.quantile(sample@x, 1-alpha)], axis=0)

def factor_expected_shortfall_contribution(factor_exposure, sample_asset_returns, beta, alpha):
    # compute the contribution of each factor to the portfolio expected shortfall
    sample_size,d = sample_asset_returns.shape
    y = cp.Variable(d)
    t = cp.Variable()
    obj = t + cp.sum(cp.maximum(-sample_asset_returns@y-t,0)/(1-alpha))/sample_size
    prob_factor = cp.Problem(
        cp.Minimize(obj),
        [beta.T@y==factor_exposure],
    )
    prob_factor.solve(verbose=False, solver='SCS')
    y_star = y.value
    pseudo_inv_beta = beta@np.linalg.inv(beta.T@beta)
    contrib_w_star = factor_exposure*(gradient_expected_shortfall(sample_asset_returns, y_star, alpha)@pseudo_inv_beta)
    return contrib_w_star


def asset_vol_contribution(x,cov):
    # compute the contribution of each asset to the portfolio volatility
    mc = grad_vol(x, cov)
    rc = x*mc
    return rc

def enb(my_weights, normalize=True):
    # compute the effective number of bets
    eff_no_bets = np.exp(-np.sum(my_weights*np.log(my_weights)))
    if normalize:
        eff_no_bets/=len(my_weights)
    return eff_no_bets

def relative_entropy(my_dist, base_dist):
    # compute the negative entropy between two distributions
    if np.any(my_dist<=0) or np.any(base_dist<=0):
        raise ValueError('Elements must be positive')
    return np.sum(my_dist*np.log(my_dist/base_dist))

def factor_risk_measure_volatility_analytical(factor_expo, matrix_y, cov):
    # compute the volatility of minimum variance portfolio satisfying factor constraints 
    y= matrix_y@factor_expo
    vol = np.sqrt(y.T@cov@y)
    return vol, y

def grad_factor_risk_measure_volatility_analytical(factor_expo, matrix_y, pseudo_inv_factor_load, cov):
    # compute the gradient of the volatility of minimum variance portfolio satisfying factor constraints
    y_star = factor_risk_measure_volatility_analytical(factor_expo, matrix_y, cov)[1]
    return grad_vol(y_star, cov)@pseudo_inv_factor_load

def diversification_ratio(x,cov):
    # compute the diversification ratio of a portfolio
    return (x@np.diag(cov))/vol(x, cov)

def fit_factor_model(df_factor_returns, df_asset_returns, p_value_threshold=0.05):
    df_betas = pd.DataFrame(columns=df_asset_returns.columns, index=df_factor_returns.columns)
    for my_asset in df_asset_returns.columns:
        X = df_factor_returns.values
        X = sm.add_constant(X)
        y = df_asset_returns[my_asset].values
        results = sm.OLS(y, X).fit()
        beta_asset = results.params[1:]*np.array(results.pvalues[1:]<p_value_threshold)
        df_betas[my_asset] = beta_asset
    df_betas = df_betas.T   
    return df_betas

def compute_risk_budgeting_portfolio(risk_budgets, cov):
    # compute the risk budgeting portfolio for given asset risk budgets
    d = len(risk_budgets)
    y = cp.Variable(d)
    obj = cp.quad_form(y, cov) - risk_budgets@cp.log(y)
    prob_factor = cp.Problem(
        cp.Minimize(obj),
        [y>=1e-10],
    )
    prob_factor.solve(verbose=False, solver='SCS')
    y_star = y.value
    theta_rb = y_star/sum(y_star)
    if prob_factor.status != 'optimal':
        print('Risk budgeting optimization did not converge')
    return y_star, theta_rb

def compute_factor_risk_budgeting_portfolio(factor_risk_budgets, betas, cov, long_only=False):
    # compute the factor risk budgeting portfolio for given factor risk budgets
    d,_ = betas.shape
    y = cp.Variable(d)
    obj = cp.quad_form(y, cov) - factor_risk_budgets@cp.log(betas.T@y)
    if long_only:
        prob_factor = cp.Problem(
            cp.Minimize(obj),
            [betas.T@y>=1e-10, y>=1e-10],
        )
    else:
        prob_factor = cp.Problem(
            cp.Minimize(obj),
            [betas.T@y>=1e-10],
        )
    prob_factor.solve(verbose=False, solver='SCS')
    y_star = y.value
    theta_frb = y_star/sum(y_star)
    if prob_factor.status != 'optimal':
        print('Factor risk budgeting optimization did not converge')
    return y_star, theta_frb

def compute_asset_factor_risk_budgeting_portfolio(asset_risk_budgets, 
                                                  factor_risk_budgets, 
                                                  asset_importance,
                                                  factor_importance,
                                                  betas,
                                                  cov):
    # compute the asset-factor risk budgeting portfolio for given asset / factor risk budgets and importance parameters
    d = len(asset_risk_budgets)
    y = cp.Variable(d)
    obj = cp.quad_form(y, cov) - factor_importance*factor_risk_budgets@cp.log(betas.T@y) - asset_importance*asset_risk_budgets@cp.log(y)
    prob_factor = cp.Problem(
        cp.Minimize(obj),
        [y>=1e-10, betas.T @ y>=1e-10],
    )
    prob_factor.solve(verbose=False, solver='SCS')
    y_star = y.value
    theta_aferb = y_star/sum(y_star)
    if prob_factor.status != 'optimal':
        print('Asset-factor risk budgeting optimization did not converge')
    return y_star, theta_aferb

def compute_minimum_variance_portfolio(cov, long_only=False):
    # compute the minimum variance portfolio
    d = cov.shape[0]
    y = cp.Variable(d)
    obj = cp.quad_form(y, cov)
    if long_only:
        prob = cp.Problem(
            cp.Minimize(obj),
            [cp.sum(y)==1, y>=0],
        )
    else:
        prob = cp.Problem(
            cp.Minimize(obj),
            [cp.sum(y)==1],
        )

    prob.solve(verbose=False, solver='SCS')
    y_star = y.value
    if prob.status != 'optimal':
        print('Minimum variance optimization did not converge')
    return y_star


def compute_nearest_asset_factor_risk_budgeting_portfolio(asset_risk_budgets, 
                                                          factor_risk_budgets, 
                                                          betas,
                                                          cov,
                                                          n_trials=200,
                                                          lambda_f = 1,
                                                          lambda_a_range = (1e-10, 5)):
    
    d, m = betas.shape
    pseudo_inv_beta = betas@np.linalg.inv(betas.T@betas)
    inv_cov = np.linalg.inv(cov)
    matrix_min_vol = inv_cov@betas@np.linalg.inv(betas.T@inv_cov@betas) 
    
    def objective(trial):
        lamdba_a = trial.suggest_float('lamdba_a', lambda_a_range[0], lambda_a_range[1])
        _, theta_aferc = compute_asset_factor_risk_budgeting_portfolio(asset_risk_budgets,
                                                                       factor_risk_budgets,
                                                                       lamdba_a,
                                                                       lambda_f,
                                                                       betas,
                                                                       cov)
        # factor allocations of the computed portfolio
        w_asset_factor_erc = theta_aferc@betas

        # Compute asset risk contributions
        vol_contrib = asset_vol_contribution(theta_aferc, cov)
        vol_contrib = vol_contrib/sum(vol_contrib)

        # Compute factor risk contributions
        factor_risk_contrib = w_asset_factor_erc*grad_factor_risk_measure_volatility_analytical(w_asset_factor_erc, matrix_min_vol, pseudo_inv_beta, cov)
        factor_risk_contrib = factor_risk_contrib/sum(factor_risk_contrib)

        return np.linalg.norm(asset_risk_budgets - vol_contrib)**2*d + np.linalg.norm(factor_risk_budgets - factor_risk_contrib)**2*m

    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials)

    _, theta_naferb  = compute_asset_factor_risk_budgeting_portfolio(asset_risk_budgets, 
                                                                     factor_risk_budgets, 
                                                                     study.best_params['lamdba_a'],
                                                                     lambda_f,
                                                                     betas,
                                                                     cov)
    
    return theta_naferb, study.best_params['lamdba_a']

def compute_expected_shortfall_risk_budgeting(sample_asset_returns, risk_budgets, alpha):
    # compute risk budgeting portfolio under expected shortfall

    sample_size,d = sample_asset_returns.shape

    y = cp.Variable(d)
    t = cp.Variable()
    obj = t + cp.sum(cp.maximum(-sample_asset_returns@y-t,0)/(1-alpha))/sample_size - risk_budgets@cp.log(y)
    prob_factor = cp.Problem(
        cp.Minimize(obj),
        [y>=1e-10],
    )
    prob_factor.solve(verbose=False, solver='SCS')
    if prob_factor.status != 'optimal':
        print('Optimization did not converge')
    theta_rb_es = y.value/sum(y.value)
    
    return y.value, theta_rb_es

def compute_expected_shortfall_factor_risk_budgeting(sample_asset_returns, beta, factor_risk_budgets, alpha, long_only=False):
    # compute factor risk budgeting portfolio under expected shortfall

    sample_size,d = sample_asset_returns.shape
    _,m = beta.shape

    y = cp.Variable(d)
    t = cp.Variable()
    obj = t + cp.sum(cp.maximum(-sample_asset_returns@y-t,0)/(1-alpha))/sample_size - factor_risk_budgets@cp.log(beta.T@y)
    if long_only:
        prob_factor = cp.Problem(
            cp.Minimize(obj),
            [y>=1e-10, beta.T@y>=1e-10],
        )
    else:
        prob_factor = cp.Problem(
            cp.Minimize(obj),
            [beta.T@y>=1e-10],
        )

    prob_factor.solve(verbose=False, solver='SCS')
    if prob_factor.status != 'optimal':
        print('Optimization did not converge')
    theta_frb_es = y.value/sum(y.value)

    return y.value, theta_frb_es

def compute_expected_shortfall_asset_factor_risk_budgeting(sample_asset_returns,
                                                           beta, 
                                                           asset_risk_budgets,
                                                           factor_risk_budgets,
                                                           asset_importance_parameter,
                                                           factor_importance_parameter,
                                                           alpha):
    # compute asset factor risk budgeting portfolio under expected shortfall

    sample_size,d = sample_asset_returns.shape
    _,m = beta.shape

    y = cp.Variable(d)
    t = cp.Variable()
    obj = t + cp.sum(cp.maximum(-sample_asset_returns@y-t,0)/(1-alpha))/sample_size - asset_importance_parameter*asset_risk_budgets@cp.log(y) - factor_importance_parameter*factor_risk_budgets@cp.log(beta.T@y)
    prob_factor = cp.Problem(
        cp.Minimize(obj),
        [y>=1e-10, beta.T@y>=1e-10],
    )

    prob_factor.solve(verbose=False, solver='SCS')
    if prob_factor.status != 'optimal':
        print('Optimization did not converge')
    theta_afrb_es = y.value/sum(y.value)

    return y.value, theta_afrb_es
