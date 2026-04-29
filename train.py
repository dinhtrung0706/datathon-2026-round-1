# @title
#!/usr/bin/env python3
"""
V7 — Stacking: LightGBM + Prophet + Ridge
==========================================
"""

import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from prophet import Prophet
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import calendar
import logging

warnings.filterwarnings("ignore")

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

DATA_DIR = "/content"
SEED = 42
# N_OPTUNA = 120
# EARLY_STOP = 200
N_OPTUNA = 1200
EARLY_STOP = 50

DROP_COLS = ["Date", "Revenue", "COGS", "gross_margin"]

TET_CENTERS = {
    2012: "2012-01-23",
    2013: "2013-02-10",
    2014: "2014-01-31",
    2015: "2015-02-19",
    2016: "2016-02-08",
    2017: "2017-01-28",
    2018: "2018-02-16",
    2019: "2019-02-05",
    2020: "2020-01-25",
    2021: "2021-02-12",
    2022: "2022-02-01",
    2023: "2023-01-22",
    2024: "2024-02-10",
}
TET_DATES = set()
for _y, _c in TET_CENTERS.items():
    _ct = pd.Timestamp(_c)
    for _d in range(-5, 6):
        TET_DATES.add((_ct + pd.Timedelta(days=_d)).date())


def is_vn_holiday(dt: pd.Timestamp) -> int:
    if dt.date() in TET_DATES:
        return 1
    m, d = dt.month, dt.day
    if (m == 4 and d == 30) or (m == 5 and d == 1):
        return 1
    if m == 9 and d in (2, 3):
        return 1
    if m == 4 and d in (18, 19, 20):
        return 1
    if m == 12 and d in (24, 25):
        return 1
    if m == 9 and d in (29, 30):
        return 1
    if m == 10 and d == 1:
        return 1
    return 0


# ═══════════════════════════════════════
# 1.  LOAD
# ═══════════════════════════════════════
print("=" * 60)
print("V7 — LightGBM + Prophet + Ridge Stacking")
print("=" * 60)

master = pd.read_csv(f"{DATA_DIR}/master_train.csv", parse_dates=["Date"])
train = pd.read_csv(f"{DATA_DIR}/Train_test.csv", parse_dates=["Date"])
val = pd.read_csv(f"{DATA_DIR}/Val_test.csv", parse_dates=["Date"])
sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv", parse_dates=["Date"])

for df in [train, val]:
    if df.columns[0] in ["Unnamed: 0", ""]:
        df.drop(df.columns[0], axis=1, inplace=True)

print(
    f"  train={train.shape[0]}, val={val.shape[0]}, master={master.shape[0]}, sub={sub.shape[0]}"
)


# ═══════════════════════════════════════
# 2.  FEATURES (V4)
# ═══════════════════════════════════════
def add_features(df):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    df["dom"] = df["Date"].dt.day
    df["days_in_month"] = df["Date"].dt.days_in_month
    df["dom_frac"] = df["dom"] / df["days_in_month"]
    df["is_eom3"] = (df["days_to_me"] <= 2).astype(int)
    df["is_som3"] = (df["dom"] <= 3).astype(int)
    for shift_col in ["COGS"]:
        for lag in [1, 7, 14, 30]:
            df[f"cogs_lag_{lag}"] = df[shift_col].shift(lag)
        df["cogs_roll_7"] = df[shift_col].shift(1).rolling(7).mean()
        df["cogs_roll_30"] = df[shift_col].shift(1).rolling(30).mean()
        df["cogs_ewm"] = df[shift_col].shift(1).ewm(alpha=0.3, adjust=False).mean()
    df["gm_lag1"] = df["gross_margin"].shift(1)
    df["gm_lag7"] = df["gross_margin"].shift(7)
    df["gm_roll7"] = df["gross_margin"].shift(1).rolling(7).mean()
    df["gm_roll30"] = df["gross_margin"].shift(1).rolling(30).mean()
    return df


master = add_features(master)
train = add_features(train)
val = add_features(val)

cogs_only = [
    "cogs_lag_1",
    "cogs_lag_7",
    "cogs_lag_14",
    "cogs_lag_30",
    "cogs_roll_7",
    "cogs_roll_30",
    "cogs_ewm",
    "gm_lag1",
    "gm_lag7",
    "gm_roll7",
    "gm_roll30",
]

rev_features = [c for c in master.columns if c not in DROP_COLS + cogs_only]
cogs_features = [c for c in master.columns if c not in DROP_COLS]

# ═══════════════════════════════════════
# 3.  PROPHET (Layer 1A)
# ═══════════════════════════════════════
print(f"\n{'=' * 60}")
print("LAYER 1A: PROPHET")
print(f"{'=' * 60}")

# Holidays
tet_centers = {
    2012: "2012-01-23",
    2013: "2013-02-10",
    2014: "2014-01-31",
    2015: "2015-02-19",
    2016: "2016-02-08",
    2017: "2017-01-28",
    2018: "2018-02-16",
    2019: "2019-02-05",
    2020: "2020-01-25",
    2021: "2021-02-12",
    2022: "2022-02-01",
    2023: "2023-01-22",
    2024: "2024-02-10",
}
hrows = []
for y, s in tet_centers.items():
    hrows.append(
        {"holiday": "tet", "ds": pd.Timestamp(s), "lower_window": -3, "upper_window": 3}
    )
for y in range(2012, 2025):
    hrows.append(
        {
            "holiday": "liberation",
            "ds": pd.Timestamp(f"{y}-04-30"),
            "lower_window": 0,
            "upper_window": 1,
        }
    )
    hrows.append(
        {
            "holiday": "national",
            "ds": pd.Timestamp(f"{y}-09-02"),
            "lower_window": 0,
            "upper_window": 1,
        }
    )
holidays_df = pd.DataFrame(hrows)

prophet_preds = {}

for target in ["Revenue", "COGS"]:
    print(f"\n  Prophet → {target}")

    # Phase 1: train on train_test → predict val
    p_train = train[["Date", target]].rename(columns={"Date": "ds", target: "y"})
    m1 = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays_df,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        holidays_prior_scale=10,
    )
    m1.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    m1.add_seasonality(name="quarterly", period=91.25, fourier_order=3)
    m1.fit(p_train)

    val_fc = m1.predict(pd.DataFrame({"ds": val["Date"]}))
    p_val = np.maximum(val_fc["yhat"].values, 0)

    mae = mean_absolute_error(val[target].values, p_val)
    rmse = np.sqrt(mean_squared_error(val[target].values, p_val))
    print(f"    Val: MAE={mae:,.0f}, RMSE={rmse:,.0f}, sum={mae + rmse:,.0f}")

    # Phase 2: train on full master → predict submission
    p_full = master[["Date", target]].rename(columns={"Date": "ds", target: "y"})
    m2 = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        holidays=holidays_df,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        holidays_prior_scale=10,
    )
    m2.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    m2.add_seasonality(name="quarterly", period=91.25, fourier_order=3)
    m2.fit(p_full)

    sub_fc = m2.predict(pd.DataFrame({"ds": sub["Date"]}))
    p_sub = np.maximum(sub_fc["yhat"].values, 0)

    prophet_preds[target] = {"val": p_val, "sub": p_sub}
    print(f"    Sub range: {p_sub.min():,.0f} – {p_sub.max():,.0f}")


# ═══════════════════════════════════════
# 4.  LIGHTGBM (Layer 1B — V4 approach)
# ═══════════════════════════════════════
print(f"\n{'=' * 60}")
print("LAYER 1B: LIGHTGBM")
print(f"{'=' * 60}")


def lgb_obj(trial, X_tr, y_tr, X_v, y_v):
    obj_type = trial.suggest_categorical(
        "objective", ["regression", "regression_l1", "huber"]
    )
    params = {
        "objective": obj_type,
        "metric": "mae",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": SEED,
        "n_jobs": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 512),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 3, 200),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 2.0),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1e-5, 100, log=True
        ),
    }
    if obj_type == "huber":
        params["huber_delta"] = trial.suggest_float("huber_delta", 0.5, 5.0)
    n_est = trial.suggest_int("n_estimators", 300, 5000)

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_v, label=y_v, reference=dtrain)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=n_est,
        valid_sets=[dval],
        callbacks=[
            lgb.early_stopping(EARLY_STOP, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    preds = np.maximum(model.predict(X_v), 0)
    return mean_absolute_error(y_v, preds) + np.sqrt(mean_squared_error(y_v, preds))


lgb_params = {}
lgb_val_preds = {}

for target in ["Revenue", "COGS"]:
    fcols = rev_features if target == "Revenue" else cogs_features
    X_tr, X_v = train[fcols], val[fcols]
    y_tr, y_v = train[target].values, val[target].values

    print(f"\n  LightGBM → {target} ({N_OPTUNA} trials)")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(
        lambda trial: lgb_obj(trial, X_tr, y_tr, X_v, y_v),
        n_trials=N_OPTUNA,
        show_progress_bar=True,
    )

    bp = study.best_params.copy()
    n_est = bp.pop("n_estimators")
    obj_type = bp.pop("objective")
    hd = bp.pop("huber_delta", None)
    p = {
        "objective": obj_type,
        "metric": "mae",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "seed": SEED,
        "n_jobs": -1,
        **bp,
    }
    if obj_type == "huber" and hd:
        p["huber_delta"] = hd
    lgb_params[target] = {"params": p, "n_est": n_est}

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval_d = lgb.Dataset(X_v, label=y_v, reference=dtrain)
    model = lgb.train(
        p,
        dtrain,
        num_boost_round=n_est,
        valid_sets=[dval_d],
        callbacks=[
            lgb.early_stopping(EARLY_STOP, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    preds = np.maximum(model.predict(X_v), 0)
    lgb_val_preds[target] = preds
    mae = mean_absolute_error(y_v, preds)
    rmse = np.sqrt(mean_squared_error(y_v, preds))
    print(
        f"    Val: MAE={mae:,.0f}, RMSE={rmse:,.0f}, sum={mae + rmse:,.0f}, obj={obj_type}"
    )


# ═══════════════════════════════════════
# 5.  RIDGE META-LEARNER (Layer 2)
# ═══════════════════════════════════════
print(f"\n{'=' * 60}")
print("LAYER 2: RIDGE META-LEARNER")
print(f"{'=' * 60}")

ridge_models = {}
for target in ["Revenue", "COGS"]:
    y_v = val[target].values
    X_meta = np.column_stack([prophet_preds[target]["val"], lgb_val_preds[target]])

    ridge = RidgeCV(alphas=np.logspace(-3, 6, 100), fit_intercept=True, cv=5)
    ridge.fit(X_meta, y_v)
    ridge_models[target] = ridge

    rp = ridge.predict(X_meta)
    mae_p = mean_absolute_error(y_v, prophet_preds[target]["val"])
    mae_l = mean_absolute_error(y_v, lgb_val_preds[target])
    mae_r = mean_absolute_error(y_v, rp)
    rmse_p = np.sqrt(mean_squared_error(y_v, prophet_preds[target]["val"]))
    rmse_l = np.sqrt(mean_squared_error(y_v, lgb_val_preds[target]))
    rmse_r = np.sqrt(mean_squared_error(y_v, rp))

    print(f"\n  {target}:")
    print(
        f"    Prophet:  MAE={mae_p:>10,.0f}  RMSE={rmse_p:>10,.0f}  sum={mae_p + rmse_p:>12,.0f}"
    )
    print(
        f"    LightGBM: MAE={mae_l:>10,.0f}  RMSE={rmse_l:>10,.0f}  sum={mae_l + rmse_l:>12,.0f}"
    )
    print(
        f"    Stacked:  MAE={mae_r:>10,.0f}  RMSE={rmse_r:>10,.0f}  sum={mae_r + rmse_r:>12,.0f}"
    )
    print(
        f"    Coefs: Prophet={ridge.coef_[0]:.4f}, LightGBM={ridge.coef_[1]:.4f}, intercept={ridge.intercept_:,.0f}"
    )

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(val.Date, y_v, label="Actual", linewidth=2, color="#333")
    ax.plot(
        val.Date,
        prophet_preds[target]["val"],
        label=f"Prophet ({mae_p:,.0f})",
        linewidth=1,
        alpha=0.5,
    )
    ax.plot(
        val.Date,
        lgb_val_preds[target],
        label=f"LightGBM ({mae_l:,.0f})",
        linewidth=1,
        alpha=0.5,
    )
    ax.plot(
        val.Date,
        rp,
        label=f"Stacked ({mae_r:,.0f})",
        linewidth=1.5,
        linestyle="--",
        color="#e74c3c",
    )
    ax.set_title(f"Stacking — {target}")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    fig.savefig(f"{DATA_DIR}/val_pred_v7_{target.lower()}.png", dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════
# 6.  RETRAIN LIGHTGBM ON FULL master
# ═══════════════════════════════════════
print(f"\n{'=' * 60}")
print("RETRAIN LIGHTGBM ON FULL DATA")
print(f"{'=' * 60}")

lgb_final = {}
for target in ["Revenue", "COGS"]:
    fcols = rev_features if target == "Revenue" else cogs_features
    cfg = lgb_params[target]
    dfull = lgb.Dataset(master[fcols], label=master[target].values)
    model = lgb.train(
        cfg["params"],
        dfull,
        num_boost_round=cfg["n_est"],
        callbacks=[lgb.log_evaluation(0)],
    )
    lgb_final[target] = model
    print(f"  {target}: trained (n_est={cfg['n_est']})")


# ═══════════════════════════════════════
# 7.  RECURSIVE PREDICTION + STACKING
# ═══════════════════════════════════════
print("\n  Generating predictions ...")

_traffic_cols = [
    "total_sessions",
    "total_visitors",
    "total_pageviews",
    "avg_bounce_rate",
    "avg_duration_sec",
    "sess_direct",
    "sess_email_campaign",
    "sess_organic_search",
    "sess_paid_search",
    "sess_referral",
    "sess_social_media",
    "paid_share",
]
_promo_cols = [
    "days_since_promo",
    "promo_active",
    "n_active_promos",
    "max_discount",
    "has_stackable",
]
_inv_cols = [
    "avg_fill_rate",
    "pct_stockout",
    "avg_stockout_days",
    "avg_sell_through",
    "pct_overstock",
    "avg_fill_rate_lag1m",
    "pct_stockout_lag1m",
    "avg_stockout_days_lag1m",
]

_m_traffic = master[["Date"] + _traffic_cols].dropna().sort_values("Date")
_m_promo = master[["Date"] + _promo_cols].sort_values("Date")
_m_inv = master[["Date"] + _inv_cols].sort_values("Date")

history = master[["Date", "Revenue", "COGS", "gross_margin"]].copy()
submission_dates = sub["Date"].tolist()
lgb_sub_rev, lgb_sub_cogs = [], []

for i, dt in enumerate(submission_dates):
    dt = pd.Timestamp(dt)
    row = {}

    row["dow"], row["month"], row["quarter"], row["year"] = (
        dt.dayofweek,
        dt.month,
        dt.quarter,
        dt.year,
    )
    row["week"] = int(dt.isocalendar()[1])
    row["is_weekend"] = 1 if dt.dayofweek >= 5 else 0
    ld = calendar.monthrange(dt.year, dt.month)[1]
    row["days_to_me"] = ld - dt.day
    row["is_month_start"] = 1 if dt.day == 1 else 0
    row["is_payday"] = 1 if dt.day in [1, 15] else 0
    row["sin_month"] = np.sin(2 * np.pi * dt.month / 12)
    row["cos_month"] = np.cos(2 * np.pi * dt.month / 12)
    row["sin_dow"] = np.sin(2 * np.pi * dt.dayofweek / 7)
    row["cos_dow"] = np.cos(2 * np.pi * dt.dayofweek / 7)
    row["dom"] = dt.day
    row["days_in_month"] = ld
    row["dom_frac"] = dt.day / ld
    row["is_eom3"] = 1 if row["days_to_me"] <= 2 else 0
    row["is_som3"] = 1 if dt.day <= 3 else 0
    row["is_holiday"] = is_vn_holiday(dt)

    rs = history.set_index("Date")["Revenue"]
    cs = history.set_index("Date")["COGS"]
    gs = history.set_index("Date")["gross_margin"]

    for lag in [1, 7, 14, 30]:
        row[f"lag_{lag}"] = rs.get(dt - pd.Timedelta(days=lag), np.nan)
    row["lag_365"] = rs.get(dt - pd.Timedelta(days=365), np.nan)
    row["lag_7d_weekday"] = rs.get(dt - pd.Timedelta(days=7), np.nan)

    rec = rs.loc[rs.index < dt].sort_index()
    n = len(rec)
    row["roll_mean_7"] = rec.iloc[-7:].mean() if n >= 7 else rec.mean()
    row["roll_mean_14"] = rec.iloc[-14:].mean() if n >= 14 else rec.mean()
    row["roll_mean_30"] = rec.iloc[-30:].mean() if n >= 30 else rec.mean()
    row["roll_std_7"] = rec.iloc[-7:].std() if n >= 7 else rec.std()
    row["roll_std_30"] = rec.iloc[-30:].std() if n >= 30 else rec.std()
    row["ewm_03"] = (
        rec.ewm(alpha=0.3, adjust=False).mean().iloc[-1] if n > 0 else np.nan
    )
    row["ewm_01"] = (
        rec.ewm(alpha=0.1, adjust=False).mean().iloc[-1] if n > 0 else np.nan
    )
    row["momentum_7"] = (
        (rec.iloc[-1] / rec.iloc[-7]) if n >= 7 and rec.iloc[-7] != 0 else 1.0
    )

    rc = cs.loc[cs.index < dt].sort_index()
    nc = len(rc)
    for lag in [1, 7, 14, 30]:
        row[f"cogs_lag_{lag}"] = cs.get(dt - pd.Timedelta(days=lag), np.nan)
    row["cogs_roll_7"] = (
        rc.iloc[-7:].mean() if nc >= 7 else (rc.mean() if nc > 0 else np.nan)
    )
    row["cogs_roll_30"] = (
        rc.iloc[-30:].mean() if nc >= 30 else (rc.mean() if nc > 0 else np.nan)
    )
    row["cogs_ewm"] = (
        rc.ewm(alpha=0.3, adjust=False).mean().iloc[-1] if nc > 0 else np.nan
    )

    rg = gs.loc[gs.index < dt].sort_index()
    ng = len(rg)
    row["gm_lag1"] = gs.get(dt - pd.Timedelta(days=1), np.nan)
    row["gm_lag7"] = gs.get(dt - pd.Timedelta(days=7), np.nan)
    row["gm_roll7"] = (
        rg.iloc[-7:].mean() if ng >= 7 else (rg.mean() if ng > 0 else np.nan)
    )
    row["gm_roll30"] = (
        rg.iloc[-30:].mean() if ng >= 30 else (rg.mean() if ng > 0 else np.nan)
    )

    tmask = _m_traffic["Date"] <= dt
    if tmask.any():
        lt = _m_traffic.loc[tmask].iloc[-1]
        for c in _traffic_cols:
            row[c] = lt[c]
    else:
        for c in _traffic_cols:
            row[c] = np.nan
    pmask = _m_promo["Date"] <= dt
    if pmask.any():
        lp = _m_promo.loc[pmask].iloc[-1]
        for c in _promo_cols:
            row[c] = lp[c]
    else:
        for c in _promo_cols:
            row[c] = 0
    imask = _m_inv["Date"] <= dt
    if imask.any():
        li = _m_inv.loc[imask].iloc[-1]
        for c in _inv_cols:
            row[c] = li[c]
    else:
        for c in _inv_cols:
            row[c] = np.nan

    rv = pd.DataFrame([{k: row.get(k, np.nan) for k in rev_features}])
    rev_lgb = max(lgb_final["Revenue"].predict(rv)[0], 0)

    cv = pd.DataFrame([{k: row.get(k, np.nan) for k in cogs_features}])
    cogs_lgb = max(lgb_final["COGS"].predict(cv)[0], 0)

    lgb_sub_rev.append(rev_lgb)
    lgb_sub_cogs.append(cogs_lgb)

    # For history: use stacked prediction
    rev_s = ridge_models["Revenue"].predict(
        np.array([[prophet_preds["Revenue"]["sub"][i], rev_lgb]])
    )[0]
    cogs_s = ridge_models["COGS"].predict(
        np.array([[prophet_preds["COGS"]["sub"][i], cogs_lgb]])
    )[0]
    rev_s = max(rev_s, 0)
    cogs_s = max(cogs_s, 0)
    if rev_s > 0:
        r = cogs_s / rev_s
        if r > 1.6:
            cogs_s = rev_s * 1.05
        elif r < 0.5:
            cogs_s = rev_s * 0.75
    gm = 1 - (cogs_s / rev_s) if rev_s > 0 else 0.15

    history = pd.concat(
        [
            history,
            pd.DataFrame(
                {
                    "Date": [dt],
                    "Revenue": [rev_s],
                    "COGS": [cogs_s],
                    "gross_margin": [gm],
                }
            ),
        ],
        ignore_index=True,
    )

    if (i + 1) % 100 == 0 or (i + 1) == len(submission_dates):
        print(f"    {i + 1}/{len(submission_dates)}")


# ═══════════════════════════════════════
# 8.  FINAL: APPLY RIDGE
# ═══════════════════════════════════════
print("\n  Applying Ridge ...")

lgb_r = np.array(lgb_sub_rev)
lgb_c = np.array(lgb_sub_cogs)
pro_r = prophet_preds["Revenue"]["sub"]
pro_c = prophet_preds["COGS"]["sub"]

final_rev = np.maximum(
    ridge_models["Revenue"].predict(np.column_stack([pro_r, lgb_r])), 0
)
final_cogs = np.maximum(
    ridge_models["COGS"].predict(np.column_stack([pro_c, lgb_c])), 0
)

for i in range(len(final_rev)):
    if final_rev[i] > 0:
        r = final_cogs[i] / final_rev[i]
        if r > 1.6:
            final_cogs[i] = final_rev[i] * 1.05
        elif r < 0.5:
            final_cogs[i] = final_rev[i] * 0.75

# ═══════════════════════════════════════
# 9.  SAVE
# ═══════════════════════════════════════
sub_out = pd.DataFrame(
    {
        "Date": sub["Date"].dt.strftime("%Y-%m-%d"),
        "Revenue": np.round(final_rev, 2),
        "COGS": np.round(final_cogs, 2),
    }
)
sub_out.to_csv(f"{DATA_DIR}/submission.csv", index=False)

# Also save pure LightGBM fallback
sub_lgb = pd.DataFrame(
    {
        "Date": sub["Date"].dt.strftime("%Y-%m-%d"),
        "Revenue": np.round(lgb_r, 2),
        "COGS": np.round(lgb_c, 2),
    }
)
sub_lgb.to_csv(f"{DATA_DIR}/submission_lgbm_only.csv", index=False)

print("\n  ✅  submission.csv (stacked)")
print(
    f"      Revenue: {sub_out['Revenue'].min():,.2f} – {sub_out['Revenue'].max():,.2f}"
)
print(f"      COGS:    {sub_out['COGS'].min():,.2f} – {sub_out['COGS'].max():,.2f}")
print("  ✅  submission_lgbm_only.csv (fallback)")

for t in ["Revenue", "COGS"]:
    lgb_final[t].save_model(f"{DATA_DIR}/lgbm_v7_{t.lower()}.txt")

print(f"\n{'=' * 60}")
print("   DONE v7! 🎉")
print(f"{'=' * 60}")

