# SFM 核心改動去掉`alpha`方法

> **一句話總結**：原論文訓練用的 $\mathcal{L}_\mu$ 有一個「每步都會自動打折縮小」的數學陷阱，導致起始時間 $t_h$ 一路縮水到 `0.10`，逼得推論時必須手動乘上 `alpha` 補救。我們先用「收益 − 代價」直接算出唯一最優起點 $\tau^*$，再把訓練目標預先「把打折補回來（除以 $\rho_h$）」，讓模型在訓練時就精準停在 $\tau^*$，從此推論不再需要 `alpha`。

---

## 一、符號

把語音生成想像成一條進度條 $t \in [0, 1]$：$t = 0$ 是雜訊，$t = 1$ 是最終清晰的真實語音頻譜 $\boldsymbol{X}_1$。

| 符號 | 白話意思 |
| :--- | :--- |
| $\boldsymbol{X}_1 = \boldsymbol{\mu}(C) + \boldsymbol{\epsilon}$ | 真實語音可拆成兩半：$\boldsymbol{\mu}(C) = \mathbb{E}[\boldsymbol{X}_1 \mid C]$ 是**看文字就能猜到的平均輪廓**（音素、大概語調）；$\boldsymbol{\epsilon} = \boldsymbol{X}_1 - \boldsymbol{\mu}(C)$ 是**看文字猜不到的隨機細節**（氣音、細微抖動）。 |
| $\hat{\boldsymbol{X}}_h$ | SFM Head 根據文字輸出的「縮放粗頻譜」。因為它只看得到文字 $C$，最多只能學到 $c \cdot \boldsymbol{\mu}(C)$（$c$ 是縮放倍率），永遠猜不到隨機的 $\boldsymbol{\epsilon}$。 |
| $t_h,\; \sigma_h$ | 粗頻譜 $\hat{\boldsymbol{X}}_h$ 在進度條上對應的**時間位置** $t_h$，以及粗頻譜本身帶有的**誤差大小** $\sigma_h$。 |
| $\|\boldsymbol{X}_1\|_m^2$ | 真實語音 $\boldsymbol{X}_1$ 扣掉 padding 後的均方值（有效格子的平均能量，即 $\|\boldsymbol{\mu}\|^2 + \|\boldsymbol{\epsilon}\|^2$）。 |
| $\rho \in (0, 1)$ | **可預測比例**（文字能解釋多少能量）：$\rho = \frac{\|\boldsymbol{\mu}\|^2}{\|\boldsymbol{\mu}\|^2 + \|\boldsymbol{\epsilon}\|^2}$。只要真實語音有猜不到的隨機細節（$\|\boldsymbol{\epsilon}\|^2 > 0$），$\rho$ 就**永遠嚴格小於 1**。 |
| $r_h = \frac{\sigma_h}{t_h}$ | **相對誤差比**（粗頻譜有多不準）。不管把粗頻譜放大縮小幾倍，分子分母同倍率消掉，所以 $r_h$ 只代表「粗頻譜品質好不好」。 |
| $1 - \sigma_{\min}$ | 幾乎等於常數 `1`（程式中 $\sigma_{\min} = 10^{-4}$，是 CondOT 終點保留的極小平滑雜訊）。 |
| $M$ | 一句頻譜扣掉補零 padding 後的**有效格子總數**（有效時間幀數 × 頻率維度數）；公式前乘 $\frac{1}{M}$ 代表平均到每個格子。 |
| $H(\cdot)$ | **微分熵（Differential Entropy）**，用來衡量機率分佈的「不確定度／雜訊混亂程度」（雜訊越大，$H$ 越大）。 |
| $D_{\text{KL}}(P \,\|\, Q)$ | **KL 散度**，衡量兩個機率分佈 $P$ 與 $Q$ 之間的差異大小（完全相同時為 `0`，偏離越多越大）。 |
| $p_0(\boldsymbol{X}_0)$ | 進度條最起點 $t = 0$ 的**標準高斯白雜訊分佈** $\mathcal{N}(\boldsymbol{0}, \boldsymbol{I})$。 |
| $p_{\text{OT}}(\boldsymbol{X}_\tau \mid \boldsymbol{X}_1)$ | 理想 CondOT 路徑在時間 $\tau$ 的**真實分佈** $\mathcal{N}\big(\tau\boldsymbol{X}_1,\, (1 - (1 - \sigma_{\min})\tau)^2\boldsymbol{I}\big)$。 |
| $p_{\text{SFM}}(\boldsymbol{X}_\tau \mid \boldsymbol{X}_1, C)$ | SFM 在時間 $\tau$ 用「粗頻譜 + 外加高斯雜訊」實際構造出來的**近似分佈** $\mathcal{N}\big(\tilde{\boldsymbol{X}}_h,\, \sigma_{\text{ext}}^2(\tau)\boldsymbol{I}\big)$。 |

---

## 二、原論文的兩大缺陷（是什麼導致必須乘 `alpha`？）

### 缺陷 1：$\mathcal{L}_\mu$ 是一個「每步都打折」的收縮陷阱
原論文用這個 MSE Loss 訓練粗頻譜 $\hat{\boldsymbol{X}}_h$：
$$\mathcal{L}_\mu = \mathbb{E}\big\|\hat{\boldsymbol{X}}_h - t_h\,\boldsymbol{X}_1\big\|^2$$
**原公式原本的意思**：既然粗頻譜投影在進度條的 $t_h$ 位置，就用 L2 誤差把它往理想點 $t_h \boldsymbol{X}_1$ 拉近。

**但這條公式會引發 3 步收縮循環**：
1. 假設目前粗頻譜是 $\hat{\boldsymbol{X}}_h = c \cdot \boldsymbol{\mu}$（倍率為 $c$）。
2. 做正交投影算時間 $t_h$（利用下方證明的 $\langle \boldsymbol{\mu}, \boldsymbol{\epsilon} \rangle = 0$）：
   $$t_h = \frac{\langle c\,\boldsymbol{\mu},\; \boldsymbol{\mu} + \boldsymbol{\epsilon} \rangle}{\|\boldsymbol{\mu} + \boldsymbol{\epsilon}\|^2} = c \cdot \underbrace{\frac{\|\boldsymbol{\mu}\|^2}{\|\boldsymbol{\mu}\|^2 + \|\boldsymbol{\epsilon}\|^2}}_{\rho \;<\; 1} = c\,\rho$$
   因為分母多了猜不到的隨機細節 $\|\boldsymbol{\epsilon}\|^2$，算出的 $t_h$ **一定比原來的 $c$ 打了個 $\rho$ 折**。
3. 接著最小化 $\mathcal{L}_\mu = \mathbb{E}\|\hat{\boldsymbol{X}}_h - t_h(\boldsymbol{\mu} + \boldsymbol{\epsilon})\|^2$：因為模型猜不到隨機的 $\boldsymbol{\epsilon}$（平均為 0），MSE 的最佳解就是輸出目標的平均值：
   $$\hat{\boldsymbol{X}}_h^{\text{new}} = t_h \cdot \boldsymbol{\mu} = (c\,\rho)\,\boldsymbol{\mu}$$

**結果**：每訓練一步，粗頻譜倍率就被乘上一個小於 1 的數 $\rho$（$c \to c\rho \to c\rho^2 \to \dots \to 0$），**唯一穩定不動點就是 0**！這導致原論文訓練完 $t_h$ 被壓到只剩 `0.10`（起點幾乎全是雜訊），只好在推論時硬乘手動倍率 `alpha = 2.5~3.5` 把它放大回去。

#### 為什麼平均輪廓 $\boldsymbol{\mu}(C)$ 和隨機細節 $\boldsymbol{\epsilon}$ 正交（內積為 `0`）？
因為 $\boldsymbol{\mu}(C) = \mathbb{E}[\boldsymbol{X}_1 \mid C]$ 是給定文字 $C$ 後的條件平均，所以殘差 $\boldsymbol{\epsilon} = \boldsymbol{X}_1 - \boldsymbol{\mu}(C)$ 在給定文字 $C$ 下的平均必然為零：
$$\mathbb{E}[\boldsymbol{\epsilon} \mid C] = \mathbb{E}[\boldsymbol{X}_1 - \boldsymbol{\mu}(C) \mid C] = \boldsymbol{\mu}(C) - \boldsymbol{\mu}(C) = \boldsymbol{0}$$
因此兩者的期望內積必然為 `0`：
$$\mathbb{E}\big[\langle \boldsymbol{\mu}(C),\, \boldsymbol{\epsilon} \rangle\big] = \mathbb{E}_C\Big[\big\langle \boldsymbol{\mu}(C),\, \underbrace{\mathbb{E}[\boldsymbol{\epsilon} \mid C]}_{=\,\boldsymbol{0}} \big\rangle\Big] = 0$$
這代表 $\boldsymbol{\mu}(C) \perp \boldsymbol{\epsilon}$（兩者正交），也因此直接成立畢氏定理：$\|\boldsymbol{X}_1\|^2 = \|\boldsymbol{\mu}(C) + \boldsymbol{\epsilon}\|^2 = \|\boldsymbol{\mu}(C)\|^2 + \|\boldsymbol{\epsilon}\|^2$。

### 缺陷 2：沒有定義「最優起點 $\tau^*$」應該停在哪裡
在進度條上選起點 $\tau$ 本質上是一個兩難權衡：
- **$\tau$ 太小（靠近 0）**：粗頻譜太弱、被外加雜訊淹沒 $\implies$ 容易念錯字（高 WER）、ODE 要走很長。
- **$\tau$ 太大（逼近上限 $\tau_{\text{sat}} = \frac{1}{(1 - \sigma_{\min}) + r_h}$）**：能補進去的外加隨機雜訊 $\sigma_{\text{ext}} = \sqrt{(1 - (1 - \sigma_{\min})\tau)^2 - r_h^2\tau^2}$ 掉到 `0` $\implies$ 沒有隨機性生成細微音色，聲音變成過平滑的機械音。

原論文沒有寫出平衡這兩端的數學公式，所以訓練不知道不縮水時到底應該停在哪裡。

---

## 三、修改：

### 第一步：用「收益 − 代價」算出唯一最優起點 $\tau^*$（解決缺陷 2）
將進度條起點 $\tau$ 的優劣寫成一條**無任何人工超參數**的淨收益函數（兩項單位皆為資訊量 nats/維）：
$$\mathcal{F}(\tau) = \underbrace{\frac{1}{M}\Big[H\big(p_0(\boldsymbol{X}_0)\big) - H\big(p_{\text{OT}}(\boldsymbol{X}_\tau \mid \boldsymbol{X}_1)\big)\Big]}_{-\ln(1 - (1 - \sigma_{\min})\tau)\;\text{（跳過 }[0,\tau]\text{ 省下的不確定度）}} \;-\; \underbrace{\frac{1}{M} D_{\text{KL}}\Big(p_{\text{SFM}}(\boldsymbol{X}_\tau \mid \boldsymbol{X}_1, C) \;\Big\|\; p_{\text{OT}}(\boldsymbol{X}_\tau \mid \boldsymbol{X}_1)\Big)}_{-\frac{1}{2}\ln\!\left(1 - \frac{r_h^2\tau^2}{(1-(1-\sigma_{\min})\tau)^2}\right)\;\text{（偏離 CondOT 的 KL 散度）}}$$

- **收益項**：$\tau$ 越大，省下的前期不確定度越多。
- **代價項**：一旦 $\tau$ 逼近上限使外加雜訊 $\sigma_{\text{ext}} \to 0$，括號內趨近 `0`，使代價 $-\ln(0) \to +\infty$，像安全氣囊一樣擋住過平滑。

對 $\mathcal{F}(\tau)$ 微分令 $\mathcal{F}'(\tau) = 0$，可解出唯一全域最優起點：
$$\tau^*(r_h) = \arg\max_{\tau \in [0, \tau_{\text{sat}})} \mathcal{F}(\tau) = \frac{2(1 - \sigma_{\min})}{2(1 - \sigma_{\min})^2 + r_h^2 + r_h\sqrt{r_h^2 + 8(1 - \sigma_{\min})^2}}$$

**公式的直覺行為**：
- 粗頻譜很準（$r_h \to 0$）$\implies \tau^* \to 1$（自動跳到終點附近）。
- 粗頻譜很爛（$r_h \to \infty$）$\implies \tau^* \to 0$（自動退回從純雜訊開始）。
- 一般 TTS（如實測 LJ Speech 的 $r_h \approx 0.92$）$\implies \tau^* \approx 0.36$（自動算出黃金折衷點）。

---

### 第二步：目標預先「除以 $\rho_h$」抵銷打折（解決缺陷 1）
既然投影時一定會被打一個 $\rho_h$ 折（$t_h = c \cdot \rho_h$），我們就在監督目標裡**預先除以 $\rho_h$ 把補回來**：
$$c^* = \frac{\tau^*(r_h)}{\rho_h}$$

接著把原論文會縮水的 $\mathcal{L}_\mu$ 換成：
$$\mathcal{L}_\mu^* = \mathbb{E}\big\|\hat{\boldsymbol{X}}_h - c^*\,\boldsymbol{X}_1\big\|_m^2$$
最小化後 $\hat{\boldsymbol{X}}_h^* = c^*\boldsymbol{\mu}(C)$，其投影時間 $t_h = c^*\rho_h = \frac{\tau^*}{\rho_h}\rho_h = \tau^*$ 鎖定在最優起點，徹底不再需要 `alpha`。

---

## 四、直接在原程式碼裡把 $t_h$ 除以 $\rho$ 就好？

**原論文直接除 $\rho$ **：

   **失去目標位置（每個點都是不動點，尺度隨便飄）**  
   若只把 $t_h$ 除以 $\rho$，代入 $\hat{\boldsymbol{X}}_h = c\boldsymbol{\mu}$ 會得到 $(t_h / \rho) = c$，於是更新後 $\hat{\boldsymbol{X}}_h^{\text{new}} = c\boldsymbol{\mu} = \hat{\boldsymbol{X}}_h$——也就是說**不管 $t_h$ 是 `0.01`、`0.10` 還是 `0.50`，通通都不會更改**！Loss 失去了把 $t_h$ 拉向最優位置的能力，訓練結果完全看隨機初始化運氣。

---

## 五、對應程式碼（僅需這幾行）

```python
# 原版（會一路往 0 縮水）
mu_loss = F.mse_loss(xp, th * x1, reduction="none")
mu_loss = torch.sum(mu_loss) / (torch.sum(mask) * x1.shape[1])

# 改動版（計算去偏最優目標 c*，徹底消除縮水與 alpha）
rho_h = torch.where(ip > 0.0, torch.clamp((ip * ip) / (xp_norm2 * p1), 0.0, 1.0 - 1e-6), torch.zeros_like(ip))
q_err = (1.0 - rho_h) * p1
c_star = ((2.0 * (1.0 - self.sigma_min)) / (
    2.0 * ((1.0 - self.sigma_min) ** 2) * rho_h
    + q_err
    + torch.sqrt(q_err * (q_err + 8.0 * ((1.0 - self.sigma_min) ** 2) * rho_h) + 1e-12)
)).to(x1.dtype)
mu_loss = F.mse_loss(xp * mask, c_star * x1 * mask, reduction="sum") / (torch.sum(mask) * n_feats)
```
