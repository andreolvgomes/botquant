def compute_hurst(resid, window=750):
    """Calcula o Expoente de Hurst (H) usando Análise R/S.
    Retorna um float entre 0 e 1.
    """

    #if len(resid) > window:
        #resid = resid[-window:]

    resid = np.asarray(resid, dtype=float)
    N = len(resid)

    # Se a série for muito curta ou sem variação, retorna 0.5 (neutro)
    if N < 20 or np.std(resid) == 0:
        return 0.5

    # Define os tamanhos de lags para a regressão R/S
    max_lag = int(N / 2)
    lags = np.unique(
        np.logspace(np.log10(2), np.log10(max_lag), num=20, dtype=int)
    )

    rs_values = []
    valid_lags = []

    for lag in lags:
        # Divide a série em blocos do tamanho 'lag'
        num_chunks = N // lag
        if num_chunks == 0:
            continue

        rs_chunk = []
        for i in range(num_chunks):
            chunk = resid[i * lag : (i + 1) * lag]

            # 1. Média do bloco
            mean = np.mean(chunk)
            # 2. Desvios acumulados em relação à média
            cum_dev = np.cumsum(chunk - mean)
            # 3. Amplitude R (Range)
            R = np.max(cum_dev) - np.min(cum_dev)
            # 4. Desvio Padrão S (Standard Deviation)
            S = np.std(chunk, ddof=1)

            if S > 0 and R > 0:
                rs_chunk.append(R / S)

        if len(rs_chunk) > 0:
            rs_values.append(np.mean(rs_chunk))
            valid_lags.append(lag)

    if len(valid_lags) < 2:
        return 0.5

    # Regressão linear no espaço log-log para extrair a inclinação (Hurst)
    poly = np.polyfit(np.log(valid_lags), np.log(rs_values), 1)

    # O coeficiente angular é o Expoente de Hurst (limitado entre 0 e 1)
    H = np.clip(poly[0], 0.0, 1.0)

    return float(H)

def compute_hurst_dfa(series, window=None):
    """
    Hurst usando DFA.
    """

    if window and len(series) > window:
        series = series[-window:]

    x = np.asarray(series, dtype=float)

    n = len(x)

    if n < 50:
        return 0.5

    x = x - x.mean()

    y = np.cumsum(x)

    max_lag = min(n // 10, 500)

    scales = np.unique(
        np.logspace(
            np.log10(8),
            np.log10(max_lag),
            num=30,
            dtype=int,
        )
    )

    fluct = []

    valid = []

    for scale in scales:

        segments = n // scale

        if segments < 2:
            continue

        rms = []

        for i in range(segments):

            seg = y[i * scale : (i + 1) * scale]

            t = np.arange(scale)

            coef = np.polyfit(t, seg, 1)

            trend = np.polyval(coef, t)

            rms.append(
                np.sqrt(np.mean((seg - trend) ** 2))
            )

        fluct.append(np.mean(rms))

        valid.append(scale)

    if len(valid) < 2:
        return 0.5

    H = np.polyfit(
        np.log(valid),
        np.log(fluct),
        1,
    )[0]

    return float(np.clip(H, 0, 1))