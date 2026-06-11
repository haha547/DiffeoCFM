# DiffeoCFM — Project Log

## 目標
以 Diffeomorphic Conditional Flow Matching (DiffeoCFM) 生成 EEG 協方差矩陣，
探討生成資料是否能改善 TD（正常發展）vs ASD（自閉症）的分類表現。

資料：自訂 EEG 資料，8 通道，兩個條件（EC=閉眼靜息、CPT=任務），43 位受測者。
預設使用 **Region S**（後 8 頻道），GroupInfo 中 S region：21 TD，22 ASD（接近平衡）。

---

## 架構概覽

### 資料標籤
- GroupInfo.mat → `condiction[1, :]`（row 1 = S region）→ 0=TD, 1=ASD
- 資料檔案格式：`G##_EC_s.npy`、`G##_CPT_s.npy`（每筆 shape: (n_trials, 8, 8)）

### 兩條實驗路線

| | Direction A | Direction B |
|--|--|--|
| 訓練腳本 | `train_custom.py` | `train_b.py` |
| 條件標籤 y | EC=0 / CPT=1 | TD-EC=0, TD-CPT=1, ASD-EC=2, ASD-CPT=3 |
| 生成器知道診斷? | 否 | 是 |
| 結果目錄 | `results/` | `results_b/` |
| ASD/TD 評估腳本 | `evaluate_a.py` | `evaluate_b.py` |

### 評估策略（LOSO）
- 外層：LeaveOneGroupOut（每次留一位 subject 作為 val）
- 內層：DiffeoCFM.fit() 內部另有 90/10 切分，僅用於 early stopping
- 評估：每個 split 的 val = 單一 subject（只有 TD 或 ASD）
  → 正確做法：**跨 split 匯集預測分數，再整體計算指標**（見 Bug 修正 #1）

### 評估指標
- **Baseline (Real→Val)**：真實訓練資料訓練分類器，測試真實 val（AugFactor=0 標記）
- **TSTR (Gen→Val)**：生成資料訓練分類器，測試真實 val（最重要，支援不同 aug 倍率）
- 分類器：TangentSpace (Riemann) + LogisticRegressionCV

### 增量生成（Augmentation）設計
- 訓練時：`--max-aug N` 預先生成 N 倍 pool 存入磁碟（預設 5）
- 評估時：`--aug k1 k2 ...` 從 pool 切片測試不同倍率，不需重新訓練
- Pool 結構：`[y[0]]*max_aug, [y[1]]*max_aug, ...`，取前 k 個即 k 倍資料
- 輸出 CSV 含 `AugFactor` 欄位，供 `plot_aug.py` 繪製趨勢圖

### 生成模型
| 方法 | Diffeomorphism | 說明 |
|------|---------------|------|
| DiffeoGauss | logeuclidean | Gaussian baseline |
| DiffeoCFM | lower_triangular | Cholesky-based CFM |
| DiffeoCFM | logeuclidean | Log-Euclidean CFM |

---

## 修改記錄

### 2026-06-03 — 初始架構理解與腳本建立

**新增/修改的檔案：**
- `train_b.py`：Direction B 訓練，4-class 聯合標籤（TD-EC/TD-CPT/ASD-EC/ASD-CPT）
- `evaluate_b.py`：Direction B 評估（原版，後來修正）
- `evaluate_a.py`：Direction A ASD/TD 評估（利用 GroupInfo 映射診斷標籤）
- `run_all.sh`：批次執行腳本

**關鍵設計決策：**
- `y = 2 * diagnosis + condition` 是將診斷與條件編碼進單一整數的方式
- 評估時解碼：`diagnosis = y // 2`，`condition = y % 2`

---

### 2026-06-03 — 修正 run_all.sh

**問題：** 腳本在 Linux 上無法執行
**原因：**
1. Linux 上 `python` 指向 Python 2，需使用 `python3`
2. `set -e` 導致一個 dataset 失敗就整個停止
3. 未偵測 data 目錄是否存在

**修正內容（`run_all.sh`）：**
- 自動偵測 `python3` 或 `python` 並驗證版本
- 移除 `set -e`，改為每步驟個別 error handling
- 加入 data 目錄存在性檢查
- 支援 `--debug`、`--region` 參數傳遞

---

### 2026-06-03 — 新增畫圖腳本 plot_asd.py

**功能：**
- 讀取 `figures/asd_classification_a.csv` 和 `figures/asd_classification_b.csv`
- 每個 dataset 產生一張獨立圖（PDF + PNG）
- Layout：rows = Condition（EC/CPT/All），cols = Direction（A/B）
- 每個 subplot：分組 bar chart，X = Method，Y = F1 or ROC-AUC
- 加入 chance level（0.5）虛線參考線

**使用：**
```bash
python plot_asd.py                  # 預設 F1
python plot_asd.py --metric roc_auc --no-trts
```

---

### 2026-06-08 — Bug 修正：LOSO 評估正確性

**Bug：** `evaluate_b.py`（及 `evaluate_a.py`）輸出「No results collected」

**根本原因：**
LOSO 每次 val 只有 1 位 subject，該 subject 只有 TD 或 ASD（單一 class）。
`clf_metrics` 檢查 `len(np.unique(y_test)) < 2` → 永遠 True → 永遠回傳 None
→ `all_rows` 永遠為空。

**修正策略（evaluate_a.py & evaluate_b.py）：**
- 舊做法（錯誤）：每個 split 直接計算 ROC-AUC/F1 → 分母為 0
- 新做法（正確）：
  1. 每個 split：訓練分類器，記錄該 subject 的 `y_score`（試次平均 P(ASD)）
  2. 全部 split 跑完後：匯集所有 (y_true, y_score)，一次計算整體指標
- 同時儲存原始預測：`figures/asd_predictions_a.csv`、`figures/asd_predictions_b.csv`

---

### 2026-06-10 — 增量生成（Augmentation Factor）支援

**需求：** 測試生成更多樣本是否能提升 TSTR 分類效果，且不需重複跑訓練。

**設計決策：**
- 生成在評估時做（不是訓練時），因為模型本身是隨機生成器，同一 y 標籤每次 sample() 結果不同
- 但模型本身訓練後不存檔，所以改成**訓練時一次生成大 pool**，評估時切片

**修改的檔案：**

| 檔案 | 修改內容 |
|------|---------|
| `train_b.py` | `--aug` 改為 `--max-aug`；run_split 生成 N×max_aug pool；存 `aug_factor_max.npy` |
| `train_custom.py` | 同上 |
| `evaluate_b.py` | 新增 `--aug k1 k2 ...`；evaluate_split 加 pool 切片邏輯；aggregate 加 AugFactor 維度 |
| `evaluate_a.py` | 同上（加 `--aug` 參數） |
| `run_all.sh` | 新增 `--max-aug`、`--aug` 參數；預設 MAX_AUG=5, AUG_TEST="1 2 3 5" |
| `plot_aug.py` | **新增**：畫 TSTR ROC-AUC / F1 vs AugFactor 趨勢圖 |

**使用流程：**
```bash
# 訓練（一次，生成 5 倍 pool）
python train_b.py --data "./cov_2s_0ov" --max-aug 5

# 評估（不需重新訓練，測試不同倍率）
python evaluate_b.py --data "./cov_2s_0ov" --aug 1 2 3 5

# 畫趨勢圖
python plot_aug.py
```

**`run_all.sh` 自訂：**
```bash
./run_all.sh --max-aug 10 --aug "1 3 5 10"
```

---

### 2026-06-11 — 修正與新功能

**修正 1：CUDA OOM（train_b.py / train_custom.py）**

**問題：** `--max-aug 5` 時，`model.sample(y_train_aug)` 一次把 N×5 個樣本丟進 `odeint` → GPU OOM。

**修正（run_split）：**
- 改成迴圈呼叫 `model.sample(y_train)` `max_aug` 次，每次只用 N 個樣本
- 結果用 `np.stack(sol_parts, axis=2).reshape(T, N*max_aug, 8, 8)` 組成正確的 pool
- Pool 排列與 `np.repeat(y_train, max_aug)` 一致，evaluate_b.py 切片邏輯不需修改

**修正 3：Availability 過濾（train_b.py / train_custom.py / evaluate_fusion.py）**

`GroupInfo.mat` 的 `availability`（3×43）欄位記錄每位受測者三種資料的可用性。
只有三行都非零的受測者才會納入訓練，共 **34 位**（排除 9 位）。

| 腳本 | 修正內容 |
|------|---------|
| `train_b.py` | 資料過濾後再加一層 `subject_available[groups]` mask |
| `train_custom.py` | 新增 `scipy.io` 載入 GroupInfo + 相同 availability mask |
| `evaluate_fusion.py` | `load_fused()` 跳過 `subject_available[sub_idx] == False` 的受測者 |

---

**修正 4：evaluate_a.py pool 切片 Bug（plot_asd Direction A 缺圖）**

**問題：** `train_custom.py --max-aug 5` 生成的 pool 有 N×5 個樣本；`evaluate_a.py` 用原始大小 N 的 mask 直接去 index (N×5, 8, 8) 陣列 → shape 不符 → 腳本崩潰 → `asd_classification_a.csv` 不存在 → `plot_asd.py` 跳過 Direction A。

**修正：** 在 `evaluate_split` 加入 pool 切片邏輯：
```python
aug_max_path = path_method / f"split_{split}_aug_factor_max.npy"
max_aug = int(np.load(aug_max_path)[0]) if aug_max_path.exists() else 1
idx_first = np.arange(0, N * max_aug, max_aug)   # [0, max_aug, 2*max_aug, ...]
gen_tr_last = gen_pool_last[idx_first]            # (N, 8, 8)
```
取每組 max_aug 個樣本中的第一個，等效於 aug_factor=1，與原始 mask shape 一致。

---

**關於 F1=0 / Baseline ROC-AUC=0.2（--debug 模式）**

這是正常的。`--debug` 只跑 2 個 LOSO splits（2 位 subjects），聚合後只有 2 個 (y_true, y_score) 資料點，任何指標都沒有統計意義。完整訓練（43 splits）才能看到有意義的結果。

---

**修正 2：LogisticRegressionCV 過慢 + sklearn FutureWarning**

**問題：** `LogisticRegressionCV(cv=5, Cs=10)` 每次 `score_subject` 跑 50 次 LR 訓練；同時觸發兩個 FutureWarning（`use_legacy_attributes`、`l1_ratios`）。

**修正（evaluate_a.py / evaluate_b.py）：**
- 換成 `LogisticRegression(C=1.0, solver="liblinear", class_weight="balanced", max_iter=1000)`
- 每次 `score_subject` 只跑 1 次，43 splits 下約快 50×
- 兩個 Warning 消失（僅 `LogisticRegressionCV` 特有）
- TSTR 評估使用固定 C=1.0 是生成模型文獻的標準做法

---

### 2026-06-11 — 新實驗：P/S Region 融合

**動機：** P（前額葉）和 S（感覺/枕葉）是兩組各 8 通道的 EEG 資料，各自生成 8×8 協方差。直接拼接成 16×16 資料量太大、訓練困難；改以數學融合方式將兩個 8×8 矩陣合成一個 8×8 矩陣，再做分類。

**新增檔案：**

| 檔案 | 說明 |
|------|------|
| `fuse.py` | 融合函式庫，含 `FUSION_METHODS` 字典，易擴充 |
| `evaluate_fusion.py` | LOSO ASD/TD 分類，輸出 `figures/fusion_classification.csv` |

**目前融合方法（`fuse.py`）：**

| 方法名 | 公式 | 說明 |
|--------|------|------|
| `arith_mean` | (P + S) / 2 | 算術平均，最快，保 SPD |
| `log_euclidean` | expm((logm(P) + logm(S)) / 2) | 幾何平均，符合 Riemannian 流形 |
| `matrix_product` | P @ S @ P | 同餘變換，以 P 的座標系描述 S，保 SPD |
| `p_only` | P | 基準：僅使用 P region |
| `s_only` | S | 基準：僅使用 S region |

**新增方法：** 在 `fuse.py` 底部的 `FUSION_METHODS` 字典加入即可。

**使用：**
```bash
python evaluate_fusion.py --data "./cov_2s_0ov"
python evaluate_fusion.py --data "./cov_2s_0ov" "./cov_4s_0ov" --methods arith_mean log_euclidean
```

**輸出：**
- `figures/fusion_predictions.csv`：每位受測者的原始預測分數
- `figures/fusion_classification.csv`：各方法的 ROC-AUC / F1 / Precision / Recall

---

## 待辦事項 / 下一步

- [x] `./run_all.sh --debug` 可以正常執行
- [x] Availability 過濾已實作（34 位有效 subjects）
- [x] F1=0 / baseline=0.2 確認為 debug 模式正常現象
- [ ] 重新以完整模式跑 `train_b.py` + `train_custom.py`（availability 過濾後結果會變）
- [ ] 重新跑 `evaluate_a.py`（pool 切片 bug 已修正，現在可以產生 asd_classification_a.csv）
- [ ] 跑 `plot_asd.py` 確認 Direction A vs B 可以正常畫圖
- [ ] 跑 `evaluate_fusion.py --data ./cov_2s_0ov` 測試 P/S 融合分類效果
- [ ] 比較融合方法 vs 單 region（p_only / s_only）的 ROC-AUC / F1
- [ ] 若融合有改善，考慮加入 DiffeoCFM 生成流程（train_fusion.py）
- [ ] 執行 `plot_aug.py` 觀察 TSTR 隨 aug 倍率的變化趨勢（完整訓練後）
