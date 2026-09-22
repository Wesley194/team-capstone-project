"""
DS-SAEA (Ma 等人, Applied Soft Computing, 2025) - 修正後的實作版本 (v2)

v2 變更說明 (數值穩定性優化)：
1. 資料標準化 (Data Normalization)：在 GPR 與 RBF 代理模型訓練前，對輸入 (X) 與輸出 (y) 進行 Z-score 標準化，以解決高維度下代理模型因為特徵與數值範圍差異過大而導致預測失真的問題。
2. 動態 RBF Epsilon (Dynamic RBF Epsilon)：取代原先寫死的 RBF_EPSILON = 1.0，改由訓練資料點之間的距離中位數動態決定，讓 RBF 網路能適應不同的目標函數縮放比例。

此實作依循論文的演算法結構：
    1. 透過 LHS 進行初始的真實評估：|D_I| = 2d
    2. 策略 1：全域基於 LCB 的 GPR 搜尋
    3. 策略 2：全域 3 個核函數的 RBF 集合搜尋
    4. 策略 3：在排名前 10% 的菁英資料上進行局部 3 個核函數的 RBF 集合搜尋
    5. Algorithm 4 的 Niche 門檻檢查同時應用於全域與局部的 RBF 搜尋
    6. JADE-ini 遵循 Algorithm 5 (最佳解 + N/4 資料點 + 6N LHS 多樣性填充)
    7. 最大的真實函數評估次數：11d

重要的重現注意事項：
- 論文指定了 Matern-3/2 GPR 的形式，並說明可透過最大似然估計 (likelihood maximization)
  來預估其參數，但並未發布針對 sigma_f、sigma_l 以及觀測雜訊的完整數值最佳化器設定。
  因此，本檔案採用了具備明確文件說明的數值邊界來進行最大似然估計。
- 論文給出了 RBF 核函數的公式，但未提供 epsilon 的具體數值。
  因此將 RBF_EPSILON 改為依資料分布動態計算。
- 論文完整的實驗使用了 20 次獨立執行與 12 個基準測試函數。
  此範例基於使用者原先的程式，目前包含了 Ellipsoid, Rosenbrock, Ackley, Griewank, Rastrigin 等函數。
- 論文並未說明 JADE 的邊界處理方式；為符合使用者原先的程式邏輯，
  此實作對於超出邊界的座標採用了隨機重新初始化的處理方式。
"""

import math
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import cdist, pdist
from scipy.stats import qmc


# ==============================================================================
# 1. PARAMETERS & CONFIGURATION
# ==============================================================================


class Config:
    """
    參數配置與全域設定 (依據論文設定)
    
    參數用途與論文對應邏輯：
    - D: 決策變數維度 (Dimension of the search space, d)
    - INIT_EVALS_FACTOR: 論文 4.1 節設定初始樣本數 |D_ini| = 2d
    - MAX_FES_FACTOR: 論文 4.1 節設定最大真實函數評估次數 MaxFEs = 11d
    - JADE_N: 論文 4.1 節設定 JADE-ini 的母體大小 N = 50
    - JADE_MAX_GEN: 論文 4.1 節設定 JADE-ini 的最大演化代數 = 50
    - JADE_P_RATE: 論文 JADE 演算法中選擇 pbest 的比例 p%
    - JADE_C: 論文 JADE 演算法的自適應參數更新率 c
    - NICHE_THRESHOLD_FACTOR: 論文 4.1 節設定的 Niche 門檻 delta = 10^-3 * sqrt(d)
    """
    # Problem / paper settings
    D = 10
    INIT_EVALS_FACTOR = 2
    MAX_FES_FACTOR = 11

    # JADE-ini settings from the paper
    JADE_N = 50
    JADE_MAX_GEN = 50

    # JADE settings
    # The paper pseudocode gives p and c as inputs and uses c=0.1 in its settings.
    # p is not given numerically in the paper text available here; 0.05 follows the
    # conventional JADE setting used in the original program.
    JADE_P_RATE = 0.05
    JADE_C = 0.1

    # RBF parameters: paper defines kernel forms but does not publish epsilon.
    RBF_EPSILON = 1.0
    RBF_JITTER = 1e-8

    # Paper parameter setting: delta = 1e-3 * sqrt(d)
    NICHE_THRESHOLD_FACTOR = 1e-3

    # GPR numerical fitting configuration.
    # Hyperparameters are fitted by maximizing the Gaussian-process marginal
    # likelihood. These bounds are implementation choices because the paper does
    # not publish the optimizer bounds/initialization.
    GPR_OPT_MAXITER = 60
    GPR_JITTER = 1e-10
    GPR_NOISE_REL_MIN = 1e-12
    GPR_NOISE_REL_MAX = 1e1
    GPR_LENGTH_REL_MIN = 1e-3
    GPR_LENGTH_REL_MAX = 1e3

    # User's original benchmark scope / run count
    RUNS = 3
    BENCHMARKS = ("Ellipsoid", "Rosenbrock", "Ackley", "Griewank", "Rastrigin")


# ==============================================================================
# 2. BENCHMARK FUNCTIONS
# ==============================================================================


class Benchmark:
    def __init__(self, name: str, d: int):
        self.name = name
        self.d = d

        # The paper's benchmark table identifies the functions, but the exact
        # numerical bounds are not part of the algorithm pseudocode. The user's
        # original program used [-5.12, 5.12], so that choice is preserved here.
        self.bounds = np.array([[-5.12, 5.12]] * d, dtype=float)

        if name == "Ellipsoid":
            self.bounds = np.array([[-5.12, 5.12]] * d, dtype=float)
        elif name == "Rastrigin":
            self.bounds = np.array([[-5.12, 5.12]] * d, dtype=float)
        elif name == "Rosenbrock":
            self.bounds = np.array([[-2.048, 2.048]] * d, dtype=float)
        elif name == "Ackley":
            self.bounds = np.array([[-32.768, 32.768]] * d, dtype=float)
        elif name == "Griewank":
            self.bounds = np.array([[-600.0, 600.0]] * d, dtype=float)
        else:
            raise ValueError(f"Unknown benchmark function: {name}")

        self.optimum = 0.0

    def evaluate(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.shape != (self.d,):
            raise ValueError(f"Expected x shape {(self.d,)}, got {x.shape}")

        if self.name == "Ellipsoid":
            i = np.arange(1, self.d + 1, dtype=float)
            return float(np.sum(i * x**2))

        if self.name == "Rastrigin":
            return float(10.0 * self.d + np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x)))

        if self.name == "Rosenbrock":
            return float(np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1.0)**2))

        if self.name == "Ackley":
            sum_sq_term = -0.2 * math.sqrt(np.mean(x**2))
            cos_term = np.mean(np.cos(2 * math.pi * x))
            return float(-20.0 * math.exp(sum_sq_term) - math.exp(cos_term) + 20.0 + math.e)

        if self.name == "Griewank":
            i = np.arange(1, self.d + 1, dtype=float)
            sum_sq = np.sum(x**2) / 4000.0
            prod_cos = np.prod(np.cos(x / np.sqrt(i)))
            return float(sum_sq - prod_cos + 1.0)

        raise RuntimeError("Unhandled benchmark")


# ==============================================================================
# 3. GPR SURROGATE
# ==============================================================================


class GPRSurrogate:
    """
    高斯過程迴歸 (Gaussian Process Regression, GPR) 代理模型
    
    演算法邏輯與論文對應 (論文 3.2 節與 Eq. 12-16)：
    - 採用 Zero-mean GP 與 Matern 3/2 核函數來建立代理模型，以逼近真實函數。
    - 藉由已知資料 D 訓練模型，推導出預測平均值 (predictive mean) 與預測變異數 (predictive variance)。
    - LCB (Least Confidence Bound) 準則：LCB_GPR(x|D) = mu_GPR(x|D) - 5 * sigma_GPR^2(x|D)
      結合了預測值與預測不確定性來找出具潛力的解。
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError("X and y have incompatible shapes")

        # --- 新增：資料標準化 (Z-score Normalization) ---
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0) + 1e-8
        self.X = (X - self.X_mean) / self.X_std

        self.y_mean = np.mean(y)
        self.y_std = np.std(y) + 1e-8
        self.y = (y - self.y_mean) / self.y_std
        # ----------------------------------------------
        
        self.n_samples, self.n_features = self.X.shape

        y_var = float(np.var(self.y))
        self.y_var = max(y_var, 1.0)
        self.sigma_f2, self.sigma_l, self.noise_var = self._fit_hyperparameters()

        self.K_XX = self._matern_kernel(self.X, self.X)
        K_reg = self.K_XX + (self.noise_var + Config.GPR_JITTER) * np.eye(self.n_samples)
        self.L = np.linalg.cholesky(K_reg)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))

    @staticmethod
    def _safe_median_distance(X: np.ndarray) -> float:
        if len(X) <= 1:
            return 1.0
        dist = pdist(X, metric="euclidean")
        positive = dist[dist > 0.0]
        if positive.size == 0:
            return 1.0
        return float(np.median(positive))

    def _matern_kernel_with_theta(
        self, X1: np.ndarray, X2: np.ndarray, sigma_f2: float, sigma_l: float
    ) -> np.ndarray:
        r = cdist(X1, X2, metric="euclidean")
        z = np.sqrt(3.0) * r / sigma_l
        return sigma_f2 * (1.0 + z) * np.exp(-z)

    def _nll(self, theta: np.ndarray) -> float:
        sigma_f2 = math.exp(float(theta[0]))
        sigma_l = math.exp(float(theta[1]))
        noise_var = math.exp(float(theta[2]))

        try:
            K = self._matern_kernel_with_theta(self.X, self.X, sigma_f2, sigma_l)
            K = K + (noise_var + Config.GPR_JITTER) * np.eye(self.n_samples)
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
            nll = (
                0.5 * float(self.y @ alpha)
                + float(np.sum(np.log(np.diag(L))))
                + 0.5 * self.n_samples * math.log(2.0 * math.pi)
            )
            if not np.isfinite(nll):
                return 1e100
            return nll
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            return 1e100

    def _fit_hyperparameters(self) -> Tuple[float, float, float]:
        y_var = self.y_var
        med_dist = self._safe_median_distance(self.X)

        sigma_f2_0 = max(float(np.var(self.y)), 1e-12)
        sigma_l_0 = max(med_dist, 1e-12)
        noise_0 = max(y_var * 1e-6, 1e-12)

        sigma_f2_min = max(y_var * 1e-6, 1e-12)
        sigma_f2_max = max(y_var * 1e6, sigma_f2_min * 10.0)
        sigma_l_min = max(med_dist * Config.GPR_LENGTH_REL_MIN, 1e-12)
        sigma_l_max = max(med_dist * Config.GPR_LENGTH_REL_MAX, sigma_l_min * 10.0)
        noise_min = max(y_var * Config.GPR_NOISE_REL_MIN, 1e-14)
        noise_max = max(y_var * Config.GPR_NOISE_REL_MAX, noise_min * 10.0)

        x0 = np.log([sigma_f2_0, sigma_l_0, noise_0])
        bounds = [
            (math.log(sigma_f2_min), math.log(sigma_f2_max)),
            (math.log(sigma_l_min), math.log(sigma_l_max)),
            (math.log(noise_min), math.log(noise_max)),
        ]

        result = minimize(
            self._nll,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": Config.GPR_OPT_MAXITER, "ftol": 1e-10},
        )

        theta = result.x if np.all(np.isfinite(result.x)) else x0
        sigma_f2, sigma_l, noise_var = np.exp(theta)
        return float(sigma_f2), float(sigma_l), float(noise_var)

    def _matern_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self._matern_kernel_with_theta(
            X1, X2, self.sigma_f2, self.sigma_l
        )

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X_test = np.asarray(X_test, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        # --- 新增：套用相同的標準化 ---
        X_test = (X_test - self.X_mean) / self.X_std
        # ---------------------------

        K_xX = self._matern_kernel(X_test, self.X)
        mu = K_xX @ self.alpha

        v = np.linalg.solve(self.L, K_xX.T)
        var = self.sigma_f2 - np.sum(v**2, axis=0)
        var = np.maximum(var, 0.0)
        
        # --- 新增：將預測值反轉換回真實尺度 ---
        mu = mu * self.y_std + self.y_mean
        var = var * (self.y_std ** 2)
        # -----------------------------------
        return mu, var

    def lcb(self, X_test: np.ndarray) -> np.ndarray:
        """Paper Eq. (16): LCB = mu_GPR - 5 * sigma_GPR^2."""
        mu, var = self.predict(X_test)
        return mu - 5.0 * var


# ==============================================================================
# 4. THREE-KERNEL RBF ENSEMBLE
# ==============================================================================


class RBFEnsemble:
    """
    徑向基函數 (Radial Basis Function, RBF) 集合代理模型
    
    演算法邏輯與論文對應 (論文 3.3 節與 Eq. 8-10, 17, 18)：
    - 建立三個不同特徵的 RBF 模型以提升預測多樣性與強健性：
      1. Multiquadric (MQ) kernel: Eq. (8)
      2. Thin plate spline (TPS) kernel: Eq. (9)
      3. Gaussian (G) kernel: Eq. (10)
    - 權重係數 w_i 藉由解線性系統 [K(X,X)]^-1 * f 求得 (Eq. 7)。
    - 集合代理預測 (Ensemble Prediction)：取三個 RBF 預測結果的最小值作為最終目標值
      F_RBF(x) = min(f_MQ(x), f_TPS(x), f_G(x)) (Eq. 17 與 Eq. 18)。
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=float).reshape(-1)
        if self.X.ndim != 2 or len(self.X) != len(self.y):
            raise ValueError("X and y have incompatible shapes")

        # --- 新增：資料標準化 (Z-score Normalization) ---
        self.X_mean = np.mean(self.X, axis=0)
        self.X_std = np.std(self.X, axis=0) + 1e-8
        self.X = (self.X - self.X_mean) / self.X_std

        self.y_mean = np.mean(self.y)
        self.y_std = np.std(self.y) + 1e-8
        self.y = (self.y - self.y_mean) / self.y_std
        # ----------------------------------------------

        # --- 新增：動態調整 Epsilon ---
        dist = pdist(self.X, metric="euclidean")
        positive_dist = dist[dist > 0.0]
        med_dist = np.median(positive_dist) if len(positive_dist) > 0 else 1.0
        self.eps = 1.0 / (med_dist + 1e-8)
        # -----------------------------

        self.jitter = Config.RBF_JITTER

        self.w_mq = self._train_weights("mq")
        self.w_tps = self._train_weights("tps")
        self.w_g = self._train_weights("gaussian")

    def _kernel_matrix(self, X1: np.ndarray, X2: np.ndarray, k_type: str) -> np.ndarray:
        r = cdist(X1, X2, metric="euclidean")
        if k_type == "mq":
            return np.sqrt(1.0 + (self.eps * r) ** 2)
        if k_type == "tps":
            return r**3
        if k_type == "gaussian":
            return np.exp(-(self.eps * r) ** 2)
        raise ValueError(f"Unknown RBF kernel: {k_type}")

    def _train_weights(self, k_type: str) -> np.ndarray:
        K = self._kernel_matrix(self.X, self.X, k_type)
        K = K + self.jitter * np.eye(K.shape[0])
        try:
            return np.linalg.solve(K, self.y)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(K, self.y, rcond=None)[0]

    def predict(self, X_test: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, ...]]:
        X_test = np.asarray(X_test, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        # --- 新增：套用相同的標準化 ---
        X_test = (X_test - self.X_mean) / self.X_std
        # ---------------------------

        f_mq = self._kernel_matrix(X_test, self.X, "mq") @ self.w_mq
        f_tps = self._kernel_matrix(X_test, self.X, "tps") @ self.w_tps
        f_g = self._kernel_matrix(X_test, self.X, "gaussian") @ self.w_g

        ensemble = np.minimum.reduce([f_mq, f_tps, f_g])
        
        # --- 新增：將預測值反轉換回真實尺度 ---
        ensemble = ensemble * self.y_std + self.y_mean
        f_mq = f_mq * self.y_std + self.y_mean
        f_tps = f_tps * self.y_std + self.y_mean
        f_g = f_g * self.y_std + self.y_mean
        # -----------------------------------
        
        return ensemble, (f_mq, f_tps, f_g)


# ==============================================================================
# 5. JADE-INI (ALGORITHM 5)
# ==============================================================================


def jade_ini(
    D_X: np.ndarray,
    D_y: np.ndarray,
    bounds: np.ndarray,
    N: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    資料驅動的族群初始化機制 (JADE-ini)
    
    演算法邏輯與論文對應 (論文 3.5 節 Algorithm 5)：
    1. 從評估過的資料庫 D 中，挑選目前最佳解放入初始母體 P_ini。
    2. 從 D 中隨機挑選共 N/4 個解加入 P_ini (包含最佳解)。
    3. 利用拉丁超立方抽樣 (LHS) 產生 6N 個候選點 (P_LHS)。
    4. 依序挑選出 3N/4 個與 P_ini 距離最遠的候選點加入 P_ini，藉此提升母體的空間分佈多樣性。
    5. 回傳大小為 N 的均勻分佈母體，作為後續 JADE 尋優的初始起點。
    """
    D_X = np.asarray(D_X, dtype=float)
    D_y = np.asarray(D_y, dtype=float)
    d = bounds.shape[0]

    if N < 4:
        raise ValueError("JADE-ini requires N >= 4")
    if len(D_X) == 0:
        raise ValueError("D cannot be empty")

    target_from_D = N // 4
    if target_from_D < 1:
        target_from_D = 1

    best_idx = int(np.argmin(D_y))
    P_ini = [D_X[best_idx].copy()]

    other = np.delete(np.arange(len(D_X)), best_idx)
    need_random = min(target_from_D - 1, len(other))
    if need_random > 0:
        chosen = rng.choice(other, size=need_random, replace=False)
        for idx in np.atleast_1d(chosen):
            P_ini.append(D_X[int(idx)].copy())

    # In normal paper settings N=50 and |D| grows beyond N/4. If D is too small,
    # duplicate-free selection from D cannot produce N/4 unique data samples.
    # We preserve uniqueness and fill the rest by the LHS diversity stage.
    P_ini_array = np.asarray(P_ini, dtype=float)

    sampler = qmc.LatinHypercube(d=d, seed=rng)
    lhs_samples = qmc.scale(
        sampler.random(n=6 * N), bounds[:, 0], bounds[:, 1]
    )

    while len(P_ini_array) < N:
        dists = cdist(lhs_samples, P_ini_array)
        min_dists = np.min(dists, axis=1)
        best_lhs_idx = int(np.argmax(min_dists))
        P_ini_array = np.vstack([P_ini_array, lhs_samples[best_lhs_idx]])
        lhs_samples = np.delete(lhs_samples, best_lhs_idx, axis=0)

    return P_ini_array[:N]


# ==============================================================================
# 6. JADE OPTIMIZER
# ==============================================================================


def _sample_cauchy_positive(mu: float, scale: float, rng: np.random.Generator) -> float:
    # JADE: F ~ Cauchy(mu_F, 0.1); resample until F > 0, then cap at 1.
    for _ in range(10000):
        u = rng.random()
        F = mu + scale * math.tan(math.pi * (u - 0.5))
        if F > 0.0:
            return min(F, 1.0)
    return min(max(mu, 1e-12), 1.0)


def _as_batch_values(
    surrogate_func: Callable[[np.ndarray], np.ndarray], X: np.ndarray
) -> np.ndarray:
    values = np.asarray(surrogate_func(X), dtype=float).reshape(-1)
    if len(values) != len(X):
        raise ValueError(
            "surrogate_func must return one scalar prediction for each input row"
        )
    return values


def jade_optimize(
    surrogate_func: Callable[[np.ndarray], np.ndarray],
    bounds: np.ndarray,
    initial_pop: np.ndarray,
    N: int,
    max_gen: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, float]:
    """
    基礎演化演算法：JADE
    
    演算法邏輯與論文對應 (論文 2.4 節 Algorithm 1)：
    - 採用 "DE/current-to-pbest/1" 突變策略 (Eq. 11)，引導解朝向目前排名前 p% 的優秀個體 (pbest) 移動。
    - 結合外部歸檔 (External Archive, A) 提供額外的多樣性，避免落入局部最佳解。
    - 交叉機率 (CR) 與突變因子 (F) 在演化過程中會進行自適應更新。
    - 本實作利用陣列批次運算評估代理模型 (surrogate_func) 尋找適應度最佳解。
    """
    bounds = np.asarray(bounds, dtype=float)
    pop = np.asarray(initial_pop, dtype=float).copy()
    d = bounds.shape[0]

    if pop.shape != (N, d):
        raise ValueError(f"Expected initial_pop shape {(N, d)}, got {pop.shape}")

    fitness = _as_batch_values(surrogate_func, pop)

    mu_CR = 0.5
    mu_F = 0.5
    archive = []
    p_rate = Config.JADE_P_RATE
    c = Config.JADE_C

    for _ in range(max_gen):
        S_CR = []
        S_F = []

        sorted_idx = np.argsort(fitness)
        num_pbest = max(1, int(p_rate * N))
        pbest_indices = sorted_idx[:num_pbest]

        CRs = np.clip(rng.normal(mu_CR, 0.1, size=N), 0.0, 1.0)
        Fs = np.array([
            _sample_cauchy_positive(mu_F, 0.1, rng) for _ in range(N)
        ])

        trials = np.empty_like(pop)

        for i in range(N):
            # pbest from current generation
            pbest_idx = int(rng.choice(pbest_indices))

            # r1 from P, excluding i
            candidates_r1 = np.delete(np.arange(N), i)
            r1 = int(rng.choice(candidates_r1))

            # r2 from P U A, excluding i and r1 when they are P members.
            total_candidates = list(range(N + len(archive)))
            total_candidates.remove(i)
            if r1 in total_candidates:
                total_candidates.remove(r1)
            r2 = int(rng.choice(total_candidates))
            x_r2 = pop[r2] if r2 < N else archive[r2 - N]

            v = (
                pop[i]
                + Fs[i] * (pop[pbest_idx] - pop[i])
                + Fs[i] * (pop[r1] - x_r2)
            )

            # The paper does not specify boundary handling; preserve the user's
            # original random reinitialization rule.
            out = (v < bounds[:, 0]) | (v > bounds[:, 1])
            if np.any(out):
                v[out] = rng.uniform(bounds[out, 0], bounds[out, 1])

            u = pop[i].copy()
            j_rand = int(rng.integers(0, d))
            take = rng.random(d) < CRs[i]
            take[j_rand] = True
            u[take] = v[take]
            trials[i] = u

        trial_fitness = _as_batch_values(surrogate_func, trials)
        success = trial_fitness < fitness

        new_pop = pop.copy()
        new_fitness = fitness.copy()

        if np.any(success):
            success_idx = np.flatnonzero(success)
            new_pop[success_idx] = trials[success_idx]
            new_fitness[success_idx] = trial_fitness[success_idx]

            for i in success_idx:
                archive.append(pop[int(i)].copy())
                S_CR.append(float(CRs[int(i)]))
                S_F.append(float(Fs[int(i)]))

        # Trim archive randomly to |A| <= N, as in Algorithm 1.
        while len(archive) > N:
            archive.pop(int(rng.integers(0, len(archive))))

        pop = new_pop
        fitness = new_fitness

        if S_CR:
            mu_CR = (1.0 - c) * mu_CR + c * float(np.mean(S_CR))
            sum_F = float(np.sum(S_F))
            if sum_F > 0.0:
                lehmer_F = float(np.sum(np.square(S_F)) / sum_F)
                mu_F = (1.0 - c) * mu_F + c * lehmer_F

    best_idx = int(np.argmin(fitness))
    return pop[best_idx].copy(), float(fitness[best_idx])


# ==============================================================================
# 7. DS-SAEA MAIN LOOP
# ==============================================================================


def _niche_ok(candidate_X: np.ndarray, D_X: np.ndarray, threshold: float) -> Tuple[bool, float]:
    min_dist = float(np.min(cdist(candidate_X.reshape(1, -1), D_X)))
    return min_dist >= threshold, min_dist


def run_ds_saea(
    bench_name: str,
    d: int,
    seed: Optional[int] = None,
    verbose: bool = True,
):
    """
    DS-SAEA 核心框架與策略切換邏輯
    
    演算法邏輯與論文對應 (論文 3.1 節 Algorithm 2)：
    1. 產生初始 LHS 樣本並透過真實昂貴函數進行評估，加入資料庫 D。
    2. 主迴圈：只要真實評估次數 (FEs) 未達上限，則交替執行三個代理模型策略。
       - Strategy 1 (全域 LCB-GPR)：利用全域資料尋找具潛力區域。若找到的解優於歷史最佳解，維持此策略。
       - Strategy 2 (全域 RBF 集合)：結合三個 RBF kernel，針對全域搜尋。若解優於歷史最佳解，維持此策略。
       - Strategy 3 (局部 RBF 集合)：僅挑選 D 中前 10% 的菁英資料 (D_e) 建立局部 RBF 模型，進行精細開發 (Exploitation)。
    3. Niche 門檻判斷 (Algorithm 4)：在 RBF 搜尋時（包含 Strategy 2 與 3），
       檢查候選解與既有資料的最短距離是否大於門檻 delta。若距離太近則拒絕評估，避免浪費資源。
    """
    rng = np.random.default_rng(seed)
    bench = Benchmark(bench_name, d)

    init_evals = Config.INIT_EVALS_FACTOR * d
    max_fes = Config.MAX_FES_FACTOR * d
    niche_threshold = Config.NICHE_THRESHOLD_FACTOR * math.sqrt(d)

    # Algorithm 2 line 1: initial LHS and true evaluation.
    sampler = qmc.LatinHypercube(d=d, seed=rng)
    lhs_samples = sampler.random(n=init_evals)
    D_X = qmc.scale(lhs_samples, bench.bounds[:, 0], bench.bounds[:, 1])
    D_y = np.array([bench.evaluate(x) for x in D_X], dtype=float)

    true_FEs = init_evals
    current_best_y = float(np.min(D_y))
    strategy = 1
    log_data = []

    if verbose:
        print(
            f"Starting {bench_name} (D={d}), MaxFEs={max_fes}, Seed={seed}"
        )
        print(f"Initial Best: {current_best_y:.6e}")

    while true_FEs < max_fes:
        best_before = current_best_y
        candidate_X = None
        surrogate_pred = np.nan
        surrogate_type = ""
        rbf_kernels_pred = None
        rejected_by_niche = False
        min_dist = np.nan
        true_y = np.nan
        strategy_before = strategy

        # Algorithm 5 is used to initialize the JADE search population for each
        # surrogate search.
        P_ini = jade_ini(D_X, D_y, bench.bounds, Config.JADE_N, rng)

        if strategy == 1:
            # ------------------------------------------------------------------
            # Strategy 1: global LCB-GPR (Algorithm 3)
            # ------------------------------------------------------------------
            surrogate_type = "LCB-GPR"
            gpr = GPRSurrogate(D_X, D_y)

            def obj_func(X):
                return gpr.lcb(X)

            candidate_X, surrogate_pred = jade_optimize(
                obj_func,
                bench.bounds,
                P_ini,
                Config.JADE_N,
                Config.JADE_MAX_GEN,
                rng,
            )

            true_y = bench.evaluate(candidate_X)
            true_FEs += 1
            D_X = np.vstack([D_X, candidate_X])
            D_y = np.append(D_y, true_y)

            # Algorithm 2: better -> strategy 1, otherwise -> strategy 2.
            if true_y < current_best_y:
                current_best_y = true_y
                strategy = 1
            else:
                strategy = 2

        elif strategy == 2:
            # ------------------------------------------------------------------
            # Strategy 2: global RBF ensemble (Algorithm 4)
            # IMPORTANT: Algorithm 4's niche test applies here too.
            # ------------------------------------------------------------------
            surrogate_type = "Global-RBF"
            rbf = RBFEnsemble(D_X, D_y)

            def obj_func(X):
                return rbf.predict(X)[0]

            candidate_X, surrogate_pred = jade_optimize(
                obj_func,
                bench.bounds,
                P_ini,
                Config.JADE_N,
                Config.JADE_MAX_GEN,
                rng,
            )

            _, kernel_preds = rbf.predict(candidate_X)
            rbf_kernels_pred = tuple(float(v[0]) for v in kernel_preds)

            accepted, min_dist = _niche_ok(candidate_X, D_X, niche_threshold)
            if accepted:
                true_y = bench.evaluate(candidate_X)
                true_FEs += 1
                D_X = np.vstack([D_X, candidate_X])
                D_y = np.append(D_y, true_y)

                # Algorithm 2: better -> strategy 2, otherwise -> strategy 3.
                if true_y < current_best_y:
                    current_best_y = true_y
                    strategy = 2
                else:
                    strategy = 3
            else:
                # Rejected by niche: no true evaluation and no database update.
                rejected_by_niche = True
                strategy = 3

        elif strategy == 3:
            # ------------------------------------------------------------------
            # Strategy 3: local RBF ensemble on elite top-10% data.
            # ------------------------------------------------------------------
            surrogate_type = "Local-RBF"
            num_elite = max(1, int(math.floor(0.10 * len(D_X))))
            elite_indices = np.argsort(D_y)[:num_elite]
            D_e_X = D_X[elite_indices]
            D_e_y = D_y[elite_indices]

            rbf_local = RBFEnsemble(D_e_X, D_e_y)

            def obj_func(X):
                return rbf_local.predict(X)[0]

            candidate_X, surrogate_pred = jade_optimize(
                obj_func,
                bench.bounds,
                P_ini,
                Config.JADE_N,
                Config.JADE_MAX_GEN,
                rng,
            )

            _, kernel_preds = rbf_local.predict(candidate_X)
            rbf_kernels_pred = tuple(float(v[0]) for v in kernel_preds)

            accepted, min_dist = _niche_ok(candidate_X, D_X, niche_threshold)
            if accepted:
                true_y = bench.evaluate(candidate_X)
                true_FEs += 1
                D_X = np.vstack([D_X, candidate_X])
                D_y = np.append(D_y, true_y)

                # Algorithm 2: better -> strategy 3, otherwise -> strategy 1.
                if true_y < current_best_y:
                    current_best_y = true_y
                    strategy = 3
                else:
                    strategy = 1
            else:
                # No true evaluation, so it cannot be a newly evaluated best.
                rejected_by_niche = True
                strategy = 1

        else:
            raise RuntimeError(f"Invalid strategy: {strategy}")

        improved = current_best_y < best_before

        log_entry = {
            "benchmark": bench_name,
            "dimension": d,
            "seed": seed,
            "true_eval_num": true_FEs,
            "strategy_before": strategy_before,
            "strategy_used": surrogate_type,
            "strategy_after": strategy,
            "candidate_pred": float(surrogate_pred),
            "true_fitness": float(true_y) if np.isfinite(true_y) else np.nan,
            "current_best_after": current_best_y,
            "improved": bool(improved),
            "rejected_by_niche": bool(rejected_by_niche),
            "niche_min_distance": min_dist,
            "niche_threshold": niche_threshold,
            "pred_mq": rbf_kernels_pred[0] if rbf_kernels_pred is not None else np.nan,
            "pred_tps": rbf_kernels_pred[1] if rbf_kernels_pred is not None else np.nan,
            "pred_gaussian": rbf_kernels_pred[2] if rbf_kernels_pred is not None else np.nan,
        }
        log_data.append(log_entry)

        if verbose:
            eval_txt = "rejected" if rejected_by_niche else f"true={true_y:.4e}"
            print(
                f"FE {true_FEs}/{max_fes} | "
                f"S{strategy_before}->{strategy} ({surrogate_type}) | "
                f"{eval_txt} | Best {current_best_y:.4e}"
            )

    best_idx = int(np.argmin(D_y))
    return log_data, current_best_y, D_X[best_idx].copy()


# ==============================================================================
# 8. MAIN EXECUTION
# ==============================================================================


def print_results_table(df: pd.DataFrame, dim: int):
    print("\n" + "="*60)
    print(f"{'Fun':<5} {'D':<5} {'DS-SAEA (Mean(STD))':<30}")
    print("-" * 60)
    
    fun_map = {
        "Ellipsoid": "F1",
        "Rosenbrock": "F2",
        "Ackley": "F3",
        "Griewank": "F4",
        "Rastrigin": "F5"
    }

    for fun in Config.BENCHMARKS:
        mask = (df['benchmark'] == fun) & (df['dimension'] == dim)
        df_fun = df[mask]
        if df_fun.empty:
            continue
            
        last_evals = df_fun.groupby('seed')['true_eval_num'].max()
        
        final_best_vals = []
        for s, max_eval in last_evals.items():
            val = df_fun[(df_fun['seed'] == s) & (df_fun['true_eval_num'] == max_eval)]['current_best_after'].values[0]
            final_best_vals.append(val)
            
        mean_val = np.mean(final_best_vals)
        std_val = np.std(final_best_vals, ddof=1) if len(final_best_vals) > 1 else 0.0
        
        formatted_str = f"{mean_val:.2E}({std_val:.2E})"
        fun_label = fun_map.get(fun, fun)
        
        print(f"{fun_label:<5} {dim:<5} {formatted_str:<30}")
    print("="*60 + "\n")


def save_summary_csv(df: pd.DataFrame, dim: int):
    """
    計算並儲存各測試函數的「平均最佳適應值 (Mean)」與「標準差 (STD)」至另一個 CSV 檔案。
    """
    summary_data = []
    
    fun_map = {
        "Ellipsoid": "F1",
        "Rosenbrock": "F2",
        "Ackley": "F3",
        "Griewank": "F4",
        "Rastrigin": "F5"
    }

    for fun in Config.BENCHMARKS:
        mask = (df['benchmark'] == fun) & (df['dimension'] == dim)
        df_fun = df[mask]
        if df_fun.empty:
            continue
            
        last_evals = df_fun.groupby('seed')['true_eval_num'].max()
        
        final_best_vals = []
        for s, max_eval in last_evals.items():
            val = df_fun[(df_fun['seed'] == s) & (df_fun['true_eval_num'] == max_eval)]['current_best_after'].values[0]
            final_best_vals.append(val)
            
        mean_val = np.mean(final_best_vals)
        std_val = np.std(final_best_vals, ddof=1) if len(final_best_vals) > 1 else 0.0
        
        fun_label = fun_map.get(fun, fun)
        
        summary_data.append({
            "Function": fun_label,
            "Benchmark": fun,
            "Dimension": dim,
            "Mean": mean_val,
            "STD": std_val,
            "Formatted (Mean(STD))": f"{mean_val:.2E}({std_val:.2E})"
        })
        
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        out_file = f"ds_saea_summary_d{dim}.csv"
        summary_df.to_csv(out_file, index=False)
        print(f"Summary data saved to {out_file}\n")


if __name__ == "__main__":
    all_logs = []
    dim = Config.D

    for bench in Config.BENCHMARKS:
        for run in range(Config.RUNS):
            seed = 42 + run
            logs, best_y, best_x = run_ds_saea(
                bench, dim, seed=seed, verbose=False
            )
            all_logs.extend(logs)
            print(
                f"--- Run {run + 1} for {bench} completed. "
                f"Best Fitness: {best_y:.6e} ---"
            )

    df = pd.DataFrame(all_logs)
    csv_file = f"ds_saea_results_d{dim}.csv"
    df.to_csv(csv_file, index=False)
    print(f"\nAll runs completed. Data saved to {csv_file}")
    
    # 輸出統整圖表至終端機
    print_results_table(df, dim)
    
    # 輸出平均與標準差統整表至獨立的 CSV
    save_summary_csv(df, dim)
