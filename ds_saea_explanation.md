# DS-SAEA 演算法運算流程與數據說明

本文件旨在詳細說明「基於高斯過程迴歸與多樣性搜尋的大型昂貴最佳化代理輔助演化演算法」(DS-SAEA) 的核心運算流程，以及程式執行後產出的實驗數據涵義。

---

## 一、 演算法運算流程 (Execution Flow)

DS-SAEA 的核心目標是為了解決「昂貴最佳化問題 (Expensive Optimization Problems, EOPs)」。這類問題的真實適應值（目標函數）計算非常耗時，因此演算法藉由建立「代理模型 (Surrogate Models)」來預測解的好壞，進而極小化真實函數的呼叫次數 (True FEs)。

演算法的整體流程如下：

### 1. 初始化階段 (Initialization)
1. **LHS 抽樣**：使用拉丁超立方抽樣 (Latin Hypercube Sampling, LHS) 在搜尋空間中均勻產生初始的候選解。
2. **真實評估**：將這些初始點帶入真實的昂貴目標函數中進行評估，並將結果存入資料庫 $D$ 中。在 $d=10$ 的設定下，初始會評估 $2d = 20$ 個點。

### 2. 代理模型輔助最佳化迴圈 (Main Loop)
程式會進入一個迴圈，直到達到最大真實評估次數 (`MaxFEs`，例如 $11d = 110$ 次) 為止。迴圈內包含了三種不同的搜尋策略 (Strategy)，根據每次搜尋的成果互相切換：

- **Strategy 1: 全域 LCB-GPR 搜尋 (Global search by LCB-based GPR)**
  - 使用資料庫 $D$ 中的 **所有** 樣本訓練高斯過程迴歸 (GPR) 模型。
  - 使用 LCB (Lower Confidence Bound) 準則來平衡探索 (Exploration，尋找未知區域) 與開發 (Exploitation，尋找當前最佳值)。
  - 若找到的解經過真實評估後，比當前歷史最佳解還要好，則繼續保持 Strategy 1；否則切換至 Strategy 2。

- **Strategy 2: 全域 RBF 集合搜尋 (Global search by RBF-based ensemble)**
  - 使用資料庫 $D$ 中的 **所有** 樣本訓練三個不同核函數 (MQ, TPS, Gaussian) 的徑向基函數網路 (RBF)。
  - 取三個 RBF 預測值的 **最小值** 作為集合代理模型的輸出。
  - 若找到的解經過真實評估後，比當前歷史最佳解還要好，則繼續保持 Strategy 2；否則切換至 Strategy 3。

- **Strategy 3: 局部 RBF 集合搜尋 (Local search by RBF-based ensemble)**
  - 僅使用資料庫 $D$ 中 **排名前 10%** 的菁英樣本 (Elite subset $D_e$) 來訓練三個不同的 RBF 模型，以針對局部最優解進行精細搜尋。
  - **Niche 門檻檢查**：為了維持多樣性並避免浪費昂貴的評估次數，只有當找出的候選解距離 $D$ 中所有已知點大於設定的門檻值 $\delta$ 時，才會進行真實評估。
  - 若被 Niche 拒絕，或真實評估後沒有打破歷史最佳紀錄，則切換回 Strategy 1；若有打破紀錄，則維持 Strategy 3。

### 3. JADE-ini 尋優
在上述每一種 Strategy 之中，當代理模型 (GPR 或 RBF) 建立好之後，需要一個內部演算法來找出該代理模型上的「最低點」。DS-SAEA 使用了改良版的 **JADE-ini**：
- 它保證了資料庫中最好的解會被放入初始母體。
- 使用距離最大化機制，從隨機生成的點中挑選出最均勻分佈的點作為 JADE 的初始族群，以提升尋優能力。

---

## 二、 輸出數據欄位說明 (Data Fields Explanation)

程式執行完畢後，會輸出一個名為 `ds_saea_results_d10.csv` 的檔案。該檔案記錄了演算法的歷史軌跡，以下是各欄位的詳細意義：

| 欄位名稱 (Column Name) | 意義與說明 |
| :--- | :--- |
| `benchmark` | 測試函數的名稱（如 `Ellipsoid` 或 `Rastrigin`）。 |
| `dimension` | 決策變數的維度（預設為 10）。 |
| `seed` | 該次獨立實驗使用的隨機亂數種子（Random Seed），用於確保實驗可重現性。 |
| `true_eval_num` | **當前真實函數評估的累積次數 (True FEs)**。這是繪製收斂曲線圖的 X 軸。 |
| `strategy_before` | 本次執行前，演算法所處的 Strategy 編號 (1, 2, 或 3)。 |
| `strategy_used` | 本次產生候選解所使用的代理模型策略（`LCB-GPR`, `Global-RBF`, `Local-RBF`）。 |
| `strategy_after` | 本次執行完畢並驗證後，演算法決定下一次要切換到的 Strategy 編號。 |
| `candidate_pred` | 代理模型對於該次候選解的 **預測適應值**（GPR 為 LCB 公式計算值；RBF 為三種 Kernel 預測值的最小值）。 |
| `true_fitness` | 該候選解帶入 **真實昂貴目標函數** 所得到的真實適應值。*(若被 Niche 門檻拒絕評估，此欄為空值)*。 |
| `current_best_after` | **歷史最佳真實適應值 (Current Best Fitness)**。即整個資料庫中適應度最低（最好）的值。這是繪製收斂曲線圖的 Y 軸。 |
| `improved` | 布林值 (`True`/`False`)。代表本次找出的解，是否成功打破了歷史最佳紀錄。 |
| `rejected_by_niche` | 布林值 (`True`/`False`)。僅在 Strategy 3 中可能為 True，代表候選解因為距離已知點太近（小於 $\delta$）而取消真實評估。 |
| `niche_min_distance` | 當次候選解與資料庫中所有已知點的最近距離。 |
| `niche_threshold` | 演算法設定的 Niche 門檻 ($\delta$)。 |
| `pred_mq` / `pred_tps` / `pred_gaussian` | 分別代表 MQ, TPS, Gaussian 三種 RBF Kernel 各自的獨立預測值（僅在 Strategy 2, 3 時有值）。 |

---

## 三、 如何利用資料重現論文圖表

1. **收斂曲線圖 (Convergence Curve)**：
   - X 軸使用 `true_eval_num`。
   - Y 軸使用 `current_best_after`（通常取 $Log_{10}$ 尺度顯示差異）。
   - 將同一個 `benchmark` 且不同 `seed` 的曲線疊加或取平均繪製，即可重現類似論文中的收斂趨勢。

2. **最佳數值統計表 (例如 Table 3)**：
   - 篩選資料表至最後一次評估，即 `true_eval_num = 110` 的那幾行。
   - 將所有 `seed` 在最後一行的 `current_best_after` 數值提取出來。
   - 計算這些數值的 **平均值 (Mean)** 與 **標準差 (STD)**，即可得到如論文表中 `5.52E-08(2.51E-08)` 格式的實驗統計結果。
