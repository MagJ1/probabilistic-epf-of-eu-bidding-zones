# src/utils/metrics.py
from typing import Sequence, List, Optional, Tuple
import torch
import numpy as np
from scipy.stats import chi2, norm

def crps_terms_slow(samples: torch.Tensor, y_true: torch.Tensor)->torch.Tensor:
    """Differentiable CRPS sample-based identity:
    CRPS ≈ (1/S)∑|Y_i - y|  -  (1/(2 S^2))∑∑|Y_i - Y_j|
    Equation according to Jordan et al. (2019), "Evaluating Probabilistic Forecasts with scoringRules". Complexity O(len(samples)^2)
    """
    # ---- CRPS estimator (all in torch; differentiable) ----
    # Term 1: (1/S) * sum_i |Y_i - y|
    # Broadcast y_true to (S, B, H) via unsqueeze
    fit = (samples - y_true.unsqueeze(0)).abs().mean(dim=0)  # (B, H)

    # Term 2: (1/(2 S^2)) * sum_{i,j} |Y_i - Y_j|
    # Pairwise absolute differences across sample dimension
    diffs = (samples.unsqueeze(0) - samples.unsqueeze(1)).abs()  # (S, S, B, H)
    spread = 0.5 * diffs.mean(dim=(0, 1))                         # (B, H)

    # CRPS per (B, H)
    # crps_bh = fit - spread  # (B, H)  # lower is better

    return fit, spread

def crps_terms_fast(samples: torch.Tensor, y_true: torch.Tensor):
    """
    Equation according to Jordan et al. (2019), "Evaluating Probabilistic Forecasts with scoringRules"
    samples: (S,B,H), y_true: (B,H)
    Returns:
      fit   = E|Y - y|           (B,H)
      spread= 0.5*E|Y - Y'|      (B,H)  via order-statistics (O(S log S))
    """
    S = samples.size(0)

    # 1) fit term: (1/S) * sum |Y_i - y|
    fit = (samples - y_true.unsqueeze(0)).abs().mean(dim=0)  # (B,H)

    # 2) spread term using sorted samples
    # Order statistics used here. \frac{1}{S^2} int_{i<j}(x_j-x_i). Thus, with sorting, the spread is just all "positive" differences between bigger and smaller values. 
    x_sorted, _ = samples.sort(dim=0)                        # (S,B,H)

    # weights w_k = 2k - S - 1 for k=1..S
    # Those weights determine how often a fixed x_k appears in the sum. As a larger element, all smaller ones are subtracted (k-1) times, resulting in +(k-1) of x_k appearances as larger element. As a smaller element, x_k appears (S-k) times as smaller element, hence -(S-k). This results in 2k-S-1.
    w = torch.arange(1, S + 1, device=samples.device, dtype=samples.dtype).view(S, 1, 1)
    w = 2 * w - S - 1                                       # (S,1,1)
    # sum_k w_k * x_(k)
    weighted = (w * x_sorted).sum(dim=0)                    # (B,H)
    spread = weighted / (S * S)                             # equals 0.5 * E|Y - Y'|  (B,H)

    return fit, spread



def es_terms_mv_beta(
    samples: torch.Tensor,
    y_true: torch.Tensor,
    beta: float,
    *,
    pair_subsample: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multivariate Energy Score terms (Monte Carlo):

      ES_beta(F, y) = E||X - y||^beta - 0.5 E||X - X'||^beta
      Complexity in O(S^2)

    Args:
      samples: (S, B, D)  samples in R^D
      y_true:  (B, D)
      beta:    in (0, 2]
      pair_subsample: optionally subsample S for the pairwise spread term

    Returns:
      fit:    (B,)  = E||X - y||^beta
      spread: (B,)  = 0.5 E||X - X'||^beta
    """
    if samples.dim() != 3:
        raise ValueError(f"es_terms_mv_beta expects samples (S,B,D), got {tuple(samples.shape)}")
    if y_true.dim() != 2:
        raise ValueError(f"es_terms_mv_beta expects y_true (B,D), got {tuple(y_true.shape)}")

    S, B, D = samples.shape
    if y_true.shape != (B, D):
        raise ValueError(f"Shape mismatch: samples (S,B,D)={tuple(samples.shape)} vs y_true={tuple(y_true.shape)}")

    # Fit term: E ||X - y||^beta
    # norm over last dim -> (S,B) -> mean over S -> (B,)
    fit = (samples - y_true.unsqueeze(0)).norm(dim=-1).pow(beta).mean(dim=0)

    # Spread term: 0.5 E ||X - X'||^beta
    X = samples
    if pair_subsample is not None and pair_subsample < S:
        idx = torch.randperm(S, device=samples.device)[:pair_subsample]
        X = X.index_select(dim=0, index=idx)  # (S',B,D)

    # pairwise differences: (S',S',B,D) -> norm over D -> (S',S',B)
    diffs = (X.unsqueeze(0) - X.unsqueeze(1)).norm(dim=-1).pow(beta)
    spread = 0.5 * diffs.mean(dim=(0, 1))  # (B,)

    return fit, spread


def sliced_es_terms_beta(
    samples: torch.Tensor,
    y_true: torch.Tensor,
    beta: float,
    *,
    K: int = 128,
    w: Optional[torch.Tensor] = None,
    pair_subsample: int | None = None,
    use_fast_for_beta1: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sliced Energy Score (Monte Carlo):
      - Draw K random unit directions w_k in R^D
      - Project X,y onto 1D: <X, w_k>, <y, w_k>
      - Compute 1D ES_beta per slice (beta=1 uses CRPS fast path)
      - Return fit/spread per (B,K)

    Args:
      samples: (S,B,D)
      y_true:  (B,D)
      beta:    in (0,2]
      K:       number of slices
      w:       optional pre-specified directions, shape (K,D)
      pair_subsample: forwarded to 1D ES for beta != 1,2
      use_fast_for_beta1: if True, uses CRPS fast path for 1D beta=1

    Returns:
      fit_bk:    (B,K)
      spread_bk: (B,K)
      w:         (K,D) used directions
    """
    if samples.dim() != 3:
        raise ValueError(f"sliced_es_terms_beta expects samples (S,B,D), got {tuple(samples.shape)}")
    if y_true.dim() != 2:
        raise ValueError(f"sliced_es_terms_beta expects y_true (B,D), got {tuple(y_true.shape)}")

    S, B, D = samples.shape
    if y_true.shape != (B, D):
        raise ValueError(f"Shape mismatch: samples={tuple(samples.shape)} vs y_true={tuple(y_true.shape)}")

    if w is None:
        w = torch.randn(K, D, device=samples.device, dtype=samples.dtype)
        w = w / w.norm(dim=1, keepdim=True).clamp_min(1e-12)  # (K,D)
    else:
        if w.dim() != 2 or w.shape[1] != D:
            raise ValueError(f"w must be (K,D) with D={D}, got {tuple(w.shape)}")
        # normalize just to be safe
        w = w / w.norm(dim=1, keepdim=True).clamp_min(1e-12)

    # projections: (S,B,D) x (K,D) -> (S,B,K)
    Yp = torch.einsum("sbd,kd->sbk", samples, w)
    yp = torch.einsum("bd,kd->bk",  y_true,  w)

    # 1D terms (vectorized across K as “last_dim”)
    fit_bk, spread_bk = crps_energy_terms_1d_beta(
        Yp, yp, beta,
        use_fast_for_beta1=use_fast_for_beta1,
        pair_subsample=pair_subsample,
    )
    return fit_bk, spread_bk, w


def sliced_energy_score(
    samples: torch.Tensor,
    y_true: torch.Tensor,
    beta: float = 1.0,
    *,
    K: int = 128,
    w: Optional[torch.Tensor] = None,
    pair_subsample: int | None = None,
    use_fast_for_beta1: bool = True,
    return_bk: bool = False,
):
    """
    Convenience wrapper returning scalar mean ES and optionally ES_{b,k} matrix.

    Returns:
      es_mean: scalar tensor
      (optional) es_bk: (B,K) tensor
      (optional) w: (K,D)
    """
    fit_bk, spread_bk, w_used = sliced_es_terms_beta(
        samples, y_true, beta,
        K=K, w=w,
        pair_subsample=pair_subsample,
        use_fast_for_beta1=use_fast_for_beta1,
    )
    es_bk = fit_bk - spread_bk  # (B,K)
    es_mean = es_bk.mean()      # scalar
    if return_bk:
        return es_mean, es_bk, fit_bk, spread_bk, w_used
    return es_mean


def crps_energy_terms_1d_beta(samples: torch.Tensor,
                         y_true: torch.Tensor,
                         beta: float,
                         *,
                         use_fast_for_beta1: bool = True,
                         pair_subsample: int | None = None):
    """

    1D Energy Score terms with power beta.
    Note: for 1D and beta=1, ES equals CRPS, so we use crps_terms_fast for speed.

    samples: (S,B,K)  or (S,B,H)  (any last axis => independent 1D problems)
    y_true:  (B,K)    or (B,H)
    Returns: (fit, spread) each shaped like y_true (B, last_dim)

    beta=2 is not recommended, as it would leave only squared errors of the predictive mean (MSE), while the variance (uncertainty) fully canceled out
    """
    if beta == 1.0 and use_fast_for_beta1:
        # Use fast CRPS: it sorts along sample axis and returns (B, last_dim)
        fit, spread = crps_terms_fast(samples, y_true)
        return fit, spread

    # Fit term for general beta: E |Y - y|^beta
    fit = (samples - y_true.unsqueeze(0)).abs().pow(beta).mean(dim=0)  # (B, last_dim)

    if beta == 2.0:
        # Variance trick: 0.5 * E|Y - Y'|^2 = Var(Y)
        mean = samples.mean(dim=0)                                      # (B, last_dim)
        var  = ((samples - mean.unsqueeze(0))**2).mean(dim=0)           # (B, last_dim)
        spread = var
        return fit, spread

    # General beta (0<beta<2, beta!=1,2): pairwise (O(S^2)) or subsampled
    if pair_subsample is not None and pair_subsample < samples.size(0):
        # Randomly pick a subset of samples for pairwise term to save compute
        idx = torch.randperm(samples.size(0), device=samples.device)[:pair_subsample]
        X = samples.index_select(dim=0, index=idx)                      # (S',B,last_dim)
    else:
        X = samples                                                     # (S,B,last_dim)

    diffs = (X.unsqueeze(0) - X.unsqueeze(1)).abs().pow(beta)           # (S',S',B,last_dim)
    spread = 0.5 * diffs.mean(dim=(0,1))                                # (B, last_dim)
    return fit, spread


def pits_from_samples(samples: np.ndarray,
                      y_true: np.ndarray,
                      randomized: bool = True,
                      rng: np.random.Generator | None = None,
                      eps: float | None = None) -> np.ndarray:
    """
    Vectorized empirical PITs for all (B, H) at once.

    Parameters
    ----------
    samples : (S, B, H) np.ndarray
        Monte Carlo draws from the predictive distribution.
    y_true  : (B, H) np.ndarray
        Realized targets.
    randomized : bool
        If True, use randomized PIT inside ties. If False, use mid-ranks (0.5).
    rng : np.random.Generator | None
        RNG for randomized ties. Default: np.random.default_rng().
    eps : float | None
        PIT clipping in (0,1) to avoid ±inf in ppf. Default: max(1e-6, 1/(S+1)).

    Returns
    -------
    U : (B, H) np.ndarray
        PITs clipped to (eps, 1-eps).
    """
    samples = np.asarray(samples, dtype=np.float64)  # CPU / float64 for stability
    y_true  = np.asarray(y_true,  dtype=np.float64)

    S, B, H = samples.shape
    X = np.moveaxis(samples, 0, -1)    # (B, H, S)
    y = y_true[..., None]              # (B, H, 1)

    # Counts along sample axis
    below = (X < y).sum(axis=-1)       # (B, H)
    equal = (X == y).sum(axis=-1)      # (B, H) -- usually 0 for continuous draws

    if randomized:
        if rng is None:
            rng = np.random.default_rng()
        frac = rng.random(size=equal.shape)    # U(0,1) per (B,H)
    else:
        frac = 0.5

    u = (below + frac * equal) / float(S)      # (B, H)

    if eps is None:
        eps = max(1e-6, 1.0 / (S + 1))         # keep away from 0/1
    return np.clip(u, eps, 1 - eps)



def berkowitz_lr_test(z: np.ndarray):
    """
    "Arg:
        z: (np.array) PITs per hour with shape (N,1)

    Returns: 
        dict with LR stat, df=3, p-value, and the MLEs.

    Explanations:
        Berkowitz (2001) LR test on normal scores z (1D array).
        Fits AR(1): z_t = mu + rho z_{t-1} + eps, eps ~ N(0, sigma^2).
        Tests H0: mu=0, rho=0, sigma^2=1 versus unrestricted.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z[np.isfinite(z)]
    if z.size < 10:
        return {"LR": np.nan, "df": 3, "p": np.nan, "T": int(z.size),
                "mu": np.nan, "rho": np.nan, "sigma2": np.nan}


    # Unrestricted AR(1) MLEs (conditional on z_0)
    z_lag = z[:-1]      # z_1,...,z_{T-1}
    z_cur = z[1:]       # z_2,...,z_T
    T1 = z_cur.size

    # Restricted log-likelihood under H0: iid N(0,1)
    # l(mu,sigma^2)=-T/2 log(2 pi sigma^2)-1/(2 sigma^2) sum_t^T (z_t-mu)^2
    ll0 = -0.5 * T1 * np.log(2 * np.pi) - 0.5 * np.sum(z_cur**2)

    # Regress z_cur on [1, z_lag] (OLS = MLE for Gaussian)
    X = np.column_stack([np.ones(T1), z_lag])        # design matrix [1, z_{t-1}], shape(T1,2)
    try:
        beta = np.linalg.lstsq(X, z_cur, rcond=None)[0]  # OLS -> MLE for Gaussian AR(1) | [mu_hat, rho_hat]
        resid = z_cur - X @ beta
        sigma2 = float(np.maximum(1e-12, np.mean(resid**2)))

        # Conditional log-likelihood (drop z_0)
        ll1 = -0.5 * T1 * (np.log(2 * np.pi * sigma2) + 1.0)

        LR = -2.0 * (ll0 - ll1)  # chi2 with 3 df (mu, rho, sigma^2)
        # p-value without scipy: use survival of chi2_3 via simple approx
        # small helper for chi2 cdf with k=3 using incomplete gamma via numpy (approx):
        p = chi2.sf(LR, df=3)  # upper-tail prob
        mu_hat, rho_hat = float(beta[0]), float(beta[1])
    except Exception:
        LR = np.nan; p = np.nan; sigma2 = np.nan; mu_hat = np.nan; rho_hat = np.nan
    
    return {
        "LR": float(LR),
        "df": 3,
        "p": p,
        "T": int(z.size),
        "mu": mu_hat,
        "rho": rho_hat,
        "sigma2": sigma2,
    }


def fisher_method(pvals, min_count=1, eps=1e-12) -> dict:
    """
    Fisher's method to combine independent p-values.
    X = -2 * sum(log(p_i)) ~ chi2(df=2k) under H0, where k = #pvals used.
    Returns dict with statistic, df, p, and k.

    Notes:
    - We clip p-values to [eps, 1-eps] to avoid infs.
    - Horizons are not strictly independent, as they originate from the same context window
    """
    pvals = [p for p in pvals if np.isfinite(p)]
    if len(pvals) < min_count:
        return {"stat": np.nan, "df": 0, "p": np.nan, "k": len(pvals)}
    pvals = np.clip(np.asarray(pvals, dtype=float), eps, 1.0 - eps)
    X = -2.0 * np.sum(np.log(pvals))
    df = 2 * len(pvals)
    p = float(chi2.sf(X, df))
    return {"stat": float(X), "df": int(df), "p": p, "k": len(pvals)}


def harmonic_mean_pvalue(pvals, eps=1e-12):
    """
    Harmonic Mean p-value (HMP). More robust under dependence than Fisher.
    Returns scalar p_HMP in (0,1]. (For small k, this is straightforward.)
    """
    p = np.asarray([pv for pv in pvals if np.isfinite(pv)], float)
    if p.size == 0:
        return np.nan
    p = np.clip(p, eps, 1.0)
    return float(p.size / np.sum(1.0 / p))



def ece_from_samples_per_horizon(
    samples: np.ndarray,  # (S,B,H)
    y_true: np.ndarray,   # (B,H)
    taus: Sequence[float],
) -> Tuple[float, List[Optional[float]], int]:
    # taus may be in percent or unit
    taus_arr = np.asarray(list(taus), dtype=float)
    if taus_arr.size == 0:
        return float("nan"), [], 0
    taus_unit = taus_arr / 100.0 if np.nanmax(taus_arr) > 1.0 else taus_arr

    S, B, H = samples.shape
    assert y_true.shape == (B, H)

    y_real_per_h: List[Optional[np.ndarray]] = []
    Q_real_per_h: List[Optional[np.ndarray]] = []

    for h in range(H):
        y_h = np.asarray(y_true[:, h], dtype=float)  # (B,)

        # quantiles across sample axis -> (T,B) then transpose -> (B,T)
        Q_h = np.quantile(samples[:, :, h], q=taus_unit, axis=0).T  # (B,T)

        # filter missing
        mask = np.isfinite(y_h) & np.all(np.isfinite(Q_h), axis=1)
        if mask.sum() == 0:
            y_real_per_h.append(None)
            Q_real_per_h.append(None)
        else:
            y_real_per_h.append(y_h[mask])
            Q_real_per_h.append(Q_h[mask])

    return compute_ece_per_h(
        y_real_per_h=y_real_per_h, Q_real_per_h=Q_real_per_h, taus=taus
    )



def compute_ece_per_h(
    *,
    y_real_per_h: List[Optional[np.ndarray]],
    Q_real_per_h: List[Optional[np.ndarray]],
    taus: Sequence[float],
) -> Tuple[float, List[Optional[float]], int]:
    """
    Quantile-calibration ECE (per horizon):
      For each tau: |mean( y <= q_tau ) - tau|
      ECE_h = mean over taus
    Returns:
      mean_ece (over horizons with data),
      ece_per_h (len=H, None if missing),
      N_used (origins per horizon, inferred from first non-empty y)
    """
    # taus may be in percent or unit
    taus_arr = np.asarray(list(taus), dtype=float)
    if taus_arr.size == 0:
        return float("nan"), [], 0
    if np.nanmax(taus_arr) > 1.0:
        taus_unit = taus_arr / 100.0
    else:
        taus_unit = taus_arr

    ece_per_h: List[Optional[float]] = []

    N_used = 0
    for y in y_real_per_h:
        if isinstance(y, np.ndarray) and y.size:
            N_used = int(len(y))
            break

    for y, Q in zip(y_real_per_h, Q_real_per_h):
        if not (isinstance(y, np.ndarray) and y.size and isinstance(Q, np.ndarray) and Q.size):
            ece_per_h.append(None)
            continue

        # Expect Q: (N, T)
        if Q.ndim != 2:
            raise ValueError(f"ECE expects Q to be 2D (N,T), got shape={Q.shape}")

        N, T = Q.shape
        if N != len(y):
            raise ValueError(f"ECE expects same N for y and Q, got len(y)={len(y)} vs Q.shape[0]={N}")

        if T != len(taus_unit):
            raise ValueError(f"ECE expects Q.shape[1]==len(taus), got {T} vs {len(taus_unit)}")

        # coverage per tau
        # (N,T) comparison -> (T,) via mean over N
        cover = (y[:, None] <= Q).mean(axis=0)  # float in [0,1]
        ece_h = float(np.mean(np.abs(cover - taus_unit)))
        ece_per_h.append(ece_h)

    vals = [v for v in ece_per_h if v is not None]
    mean_ece = float(np.mean(vals)) if vals else float("nan")
    return mean_ece, ece_per_h, N_used