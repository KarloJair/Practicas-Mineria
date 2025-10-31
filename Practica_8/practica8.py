import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def ensure_outputs():
    out = "Practica_8/outputs"
    os.makedirs(out, exist_ok=True)
    return out

def load_and_clean(path, max_year_allowed=2016):
    print(f"Cargando dataset desde '{path}' ...")
    df = pd.read_csv(path)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    before = len(df)
    df['Year'] = df['Year'].astype(int)
    after = len(df)
    df = df[df['Year'] <= max_year_allowed].copy()
    after_filter = len(df)


    print(f"Filas totales: {before}. Tras eliminar nulos Year: {after}. Tras filtrar Year<={max_year_allowed}: {after_filter}.")
    if after_filter == 0:
        raise ValueError("No hay datos después del filtrado por año.")
    return df

def aggregate_by_year(df):
    agg = df.groupby('Year', as_index=False)['Global_Sales'].sum().sort_values('Year').reset_index(drop=True)
    return agg

def winsorize_series(series, lower_q=0.01, upper_q=0.99):
    low = series.quantile(lower_q)
    high = series.quantile(upper_q)
    return series.clip(low, high)

def create_lag_features(agg, nlags=3):
    df = agg.copy()
    for lag in range(1, nlags+1):
        df[f'lag{lag}'] = df['Global_Sales'].shift(lag)
    df = df.dropna().reset_index(drop=True)
    return df

def plot_timeseries(agg, outdir):
    plt.figure(figsize=(10,5))
    plt.plot(agg['Year'], agg['Global_Sales'], marker='o')
    plt.title('Ventas globales por año (suma anual)')
    plt.xlabel('Año'); plt.ylabel('Global_Sales'); plt.grid(True)
    fname = os.path.join(outdir, 'ts_total_sales.png')
    plt.savefig(fname, bbox_inches='tight'); plt.close()

def train_with_timeseries_cv(df_features, target_col='y', n_splits=4, alphas=[0.01,0.1,1.0,10.0]):
    X = df_features.drop(columns=['Year', target_col]).values
    y = df_features[target_col].values
    tscv = TimeSeriesSplit(n_splits=min(n_splits, max(1, len(df_features)-2)))
    best_alpha = None
    best_score = np.inf

    for alpha in alphas:
        mses = []
        for train_idx, val_idx in tscv.split(X):
            Xtr, Xv = X[train_idx], X[val_idx]
            ytr, yv = y[train_idx], y[val_idx]
            scaler = StandardScaler().fit(Xtr)
            Xtr_s = scaler.transform(Xtr)
            Xv_s = scaler.transform(Xv)
            model = Ridge(alpha=alpha)
            model.fit(Xtr_s, ytr)
            ypv = model.predict(Xv_s)
            mses.append(mean_squared_error(yv, ypv))
        avg = np.mean(mses)
        if avg < best_score:
            best_score = avg
            best_alpha = alpha

    scaler_full = StandardScaler().fit(X)
    Xs = scaler_full.transform(X)
    model_final = Ridge(alpha=best_alpha).fit(Xs, y)

    return {'model': model_final, 'scaler': scaler_full, 'alpha': best_alpha}

def walk_forward_evaluate(df_features, model_info, target_col='y', test_horizon=3):
    years = df_features['Year'].values
    X_all = df_features.drop(columns=['Year', target_col])
    y_all = df_features[target_col].values

    scaler = model_info['scaler']
    model = model_info['model']

    preds = []
    actuals = []
    pred_years = []

    n = len(df_features)
    start = n - test_horizon
    if start < 2:
        start = 2  # mínimo
    for i in range(start, n):
        # train on 0..i-1, predict i
        X_train = X_all.iloc[:i].values
        y_train = y_all[:i]
        X_test = X_all.iloc[i].values.reshape(1, -1)
        # re-fit scaler & model on train
        scaler_t = StandardScaler().fit(X_train)
        Xtr_s = scaler_t.transform(X_train)
        model_t = Ridge(alpha=model.alpha).fit(Xtr_s, y_train)
        Xtest_s = scaler_t.transform(X_test)
        ypred = model_t.predict(Xtest_s)[0]
        preds.append(ypred)
        actuals.append(y_all[i])
        pred_years.append(int(years[i]))

    return pd.DataFrame({'Year': pred_years, 'Pred_trans': preds, 'Actual_trans': actuals})

def invert_transform(series_trans, transform):
    if transform == 'log':
        return np.expm1(np.array(series_trans))
    else:
        return np.array(series_trans)

def forecast_iterative(agg_orig, last_known_row, model_info, forecast_years=5, transform='log'):
    nlags = sum(1 for c in last_known_row.index if c.startswith('lag'))

    last_year = int(agg_orig['Year'].max())
    recent = agg_orig.sort_values('Year', ascending=True).tail(nlags)['Global_Sales'].values
    preds = []
    current_lags = recent.copy()
    for i in range(1, forecast_years+1):
        year = last_year + i

        feat = current_lags[::-1]
        feat = feat.reshape(1, -1)
        scaler = model_info['scaler']
        feat_s = scaler.transform(feat)
        model = model_info['model']
        pred_trans = model.predict(feat_s)[0]

        pred_orig = invert_transform(pred_trans, transform)
        preds.append((year, pred_orig))
        current_lags = np.concatenate(([pred_orig], current_lags[:-1]))
    return pd.DataFrame(preds, columns=['Year','Forecast_Global_Sales'])

def metrics_on_original(pred_df_trans, invert_transform_fn, transform):
    pred_orig = invert_transform(pred_df_trans['Pred_trans'].values, transform)
    act_orig = invert_transform(pred_df_trans['Actual_trans'].values, transform)
    mse = mean_squared_error(act_orig, pred_orig)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(act_orig, pred_orig))

    denom = np.where(np.abs(act_orig) < 1e-6, 1e-6, np.abs(act_orig))
    mape = float(np.mean(np.abs((act_orig - pred_orig) / denom))) * 100.0
    return {'RMSE': rmse, 'MAE': mae, 'MAPE(%)': mape, 'pred_orig': pred_orig, 'act_orig': act_orig}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='vgsales.csv')
    parser.add_argument('--forecast_years', '-f', type=int, default=5)
    parser.add_argument('--max_year', type=int, default=2012)
    parser.add_argument('--transform', choices=['none','log'], default='log', help='Transformación para estabilizar varianza (log recomendado)')
    args = parser.parse_args()

    outdir = ensure_outputs()

    df = load_and_clean(args.input, max_year_allowed=args.max_year)
    agg = aggregate_by_year(df)
    print(f"Años en los datos agregados: {agg['Year'].min()} - {agg['Year'].max()} (N={len(agg)})")
    plot_timeseries(agg, outdir)

    agg['Global_Sales_w'] = winsorize_series(agg['Global_Sales'], 0.01, 0.99)

    # Transform target
    transform = args.transform
    if transform == 'log':
        agg['y'] = np.log1p(agg['Global_Sales_w'])
    else:
        agg['y'] = agg['Global_Sales_w'].values

    df_feat = create_lag_features(agg[['Year','Global_Sales_w','y']].rename(columns={'Global_Sales_w':'Global_Sales'}), nlags=3)
    df_final = df_feat[['Year','lag1','lag2','lag3','y']].copy()
    df_final = df_final.sort_values('Year').reset_index(drop=True)

    # Entrenamiento con TimeSeries CV y Ridge
    model_info = train_with_timeseries_cv(df_final.rename(columns={'y':'y'}), target_col='y', n_splits=4, alphas=[0.01,0.1,1.0,10.0])
    print(f"Alpha elegido por CV: {model_info['model'].alpha}")

    test_horizon = 3
    wf_df = walk_forward_evaluate(df_final.rename(columns={'y':'y'}), model_info, target_col='y', test_horizon=test_horizon)

    # Métricas en escala original
    m = metrics_on_original(wf_df, invert_transform, transform)
    print("Evaluación walk-forward (últimos años):")
    print(f" RMSE = {m['RMSE']:.4f}, MAE = {m['MAE']:.4f}, MAPE = {m['MAPE(%)']:.2f}%")

    df_cmp = pd.DataFrame({'Year': wf_df['Year'], 'Actual': m['act_orig'], 'Predicted': m['pred_orig']})
    cmp_path = os.path.join(outdir, 'pred_vs_actual_walkforward.csv')
    df_cmp.to_csv(cmp_path, index=False)
    # Plot
    plt.figure(figsize=(8,4))
    plt.plot(df_cmp['Year'], df_cmp['Actual'], marker='o', label='Actual')
    plt.plot(df_cmp['Year'], df_cmp['Predicted'], marker='o', linestyle='--', label='Predicho')
    plt.title('Walk-forward: Actual vs Predicho (escala original)')
    plt.xlabel('Año'); plt.ylabel('Global_Sales'); plt.legend(); plt.grid(True)
    ppath = os.path.join(outdir, 'pred_vs_actual_walkforward.png')
    plt.savefig(ppath, bbox_inches='tight'); plt.close()

    X_all = df_final[['lag1','lag2','lag3']].values
    y_all = df_final['y'].values
    scaler_full = StandardScaler().fit(X_all)
    Xs = scaler_full.transform(X_all)
    model_final = Ridge(alpha=model_info['model'].alpha).fit(Xs, y_all)
    model_info_full = {'model': model_final, 'scaler': scaler_full}

    df_forecast = forecast_iterative(agg_orig=agg[['Year','Global_Sales']], last_known_row=df_final.iloc[-1], model_info=model_info_full, forecast_years=args.forecast_years, transform=transform)
    csvf = os.path.join(outdir, 'forecast.csv')
    df_forecast.to_csv(csvf, index=False)

    plt.figure(figsize=(10,5))
    plt.plot(agg['Year'], agg['Global_Sales'], marker='o', label='Histórico (original)')
    plt.plot(df_forecast['Year'], df_forecast['Forecast_Global_Sales'], marker='o', linestyle='--', label='Forecast')
    plt.title('Ventas globales: histórico y forecast (original scale)')
    plt.xlabel('Año'); plt.ylabel('Global_Sales'); plt.grid(True); plt.legend()
    fname = os.path.join(outdir, 'forecast_plot.png')
    plt.savefig(fname, bbox_inches='tight'); plt.close()


if __name__ == '__main__':
    main()
