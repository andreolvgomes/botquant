
def run_pykalman(y, x, delta=1e-5):
    # ajusta o formato da matriz de observação [x, 1]
    obs_mat = np.vstack([x.values, np.ones(len(x))]).T[:, np.newaxis, :]

    # configura a variância de transição do estado (Q)
    trans_cov = (delta / (1 - delta)) * np.eye(2)

    # inicializa o Filtro de Kalman
    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=2,
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.ones((2, 2)),
        transition_matrices=np.eye(2),
        observation_matrices=obs_mat,
        observation_covariance=1.0,
        transition_covariance=trans_cov,
    )

    # roda o filtro para obter os estados históricos (Beta e Alpha)
    state_means, _ = kf.filter(y.values)
    
    beta = state_means[:, 0]
    alpha = state_means[:, 1]

    # calcula o resíduo do modelo
    residuo = y.values - (beta * x.values + alpha)

    #return residuo
    return pd.DataFrame(
        {"beta": beta, "alpha": alpha, "resid": residuo}, index=y.index
    )
