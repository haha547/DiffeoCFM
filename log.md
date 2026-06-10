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
- **Baseline (Real→Val)**：真實訓練資料訓練分類器，測試真實 val
- **TSTR (Gen→Val)**：生成資料訓練分類器，測試真實 val（最重要）
- 分類器：TangentSpace (Riemann) + LogisticRegressionCV

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

## 待辦事項 / 下一步

- [ ] 在 Linux 主機上實際執行 `train_b.py` 並驗證輸出
- [ ] 執行 `evaluate_b.py` 確認 bug 已修正
- [ ] 執行 `plot_asd.py` 產生結果圖
- [ ] 比較 Direction A vs B 在各 dataset 的 TSTR F1
- [ ] 確認 `availability` 欄位是否需要用來過濾缺失受測者
