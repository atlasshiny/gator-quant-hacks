# Risk Management (Rough Draft)

*Implementation: [`risk_manager.py`](risk_manager.py). Values marked [sim] were chosen by simulation on **assumed** model behavior and will be re-tuned once the model is trained.*

Our goal is to survive bad periods without giving up the edge in good ones. The system has two layers. A **general layer** holds the controls any systematic strategy should run. A **strategy-specific layer** reflects our main risk: an adaptive GA + XGBoost model can **stop working without warning** as features rotate and regimes change, so we treat model health as a risk input alongside price risk.

At each rebalance, every rule outputs a scale: 1 = trade normally, 0.5 = half size, 0 = flat. The most cautious scale wins. We then size positions, remove factor bets, scale to the volatility target, and apply every limit. Finally, we re-price the book on past days and stress scenarios, and any limit breach blocks the trade.

## Risk Controls

| Control | Setting | Why |
|---|---|---|
| **Position sizing** | $w_i \propto s_i/\sigma_i$ | Size by risk: signal ÷ volatility. |
| **Volatility target** | 10%/yr, leverage ≤ 4× | Scale $k=\sigma^*/\sqrt{252\,w^\top\Sigma w}$ so risk stays constant. EWMA covariance, 30% shrinkage. It acts as a ceiling: position caps often bind first. |
| **Position limit** | 2% of NAV; ≤ 1% of ADV | A small edge must be spread across many names. Staying tiny relative to daily volume keeps our 2–5 bps cost assumption valid. |
| **Gross / net exposure** | ≤ 150% / ±10% | Market-neutral. Gross is capped because high turnover multiplies costs. |
| **Minimum signal** | \|s\| ≥ 0.02 | Signals weaker than this don't beat costs. |
| **Drawdown rule** [sim] | −5% → half size; −15% → flat 10 days, then reset the peak | Flattening at −8% or −10% sold the bottom too often. The cooldown leaves time to re-validate the model. |
| **Daily loss** [sim] | −1.5% → stop for the day | −1% stopped on noise; −3% came too late. |
| **Signal decay** [sim] | 20-day IC ≤ −0.01 → half size | IC (prediction vs. outcome rank correlation) is noisy, so act only when it's clearly negative. |
| **Feature-mask stability** [sim] | Monitor only | Jaccard overlap $J=\lvert M_t\cap M_{t-1}\rvert/\lvert M_t\cup M_{t-1}\rvert$. Cutting size on feature swaps produced more false alarms than saved losses. |
| **Per-trade stops** [sim] | 3σ of the horizon move; exit after 1.5× horizon (1× in high vol) | Tight stops cut good trades on noise. The time stop matters more, because the model only predicts ~1 horizon ahead. |
| **Kill switches** | Data > 60 s old; prediction outside training range | Bad inputs still produce confident-looking outputs. |

## Risk Decomposition

**Factor and correlation exposure.** A GA searching thousands of features can rediscover momentum or reversal and present it as alpha. Each rebalance we project out market, size, momentum and value exposure ($w \leftarrow w - X(X^\top X)^{-1}X^\top w$), with limits of |β| ≤ 0.10 and style factors ≤ 0.20. Any sector or correlation cluster (ρ > 0.7) is capped at 20% of gross. We report each asset's share of risk ($w_i(\Sigma w)_i / w^\top\Sigma w$) and the factor vs. specific split, targeting **≥ 90% specific risk and ≤ 10% of risk per asset**. Backtest returns are reported before and after removing factor returns.

**Tail and regime risk.**
- **Volatility regime:** we compare 20-day with 252-day *market* volatility. At 1.5× we cut to half size and at 2.5× we go flat [sim]. We use the market's volatility because our own volatility targeting hides the spike in our returns.
- **Expected Shortfall:** forward 97.5% ES (the average loss on the worst 2.5% of days), measured by re-pricing today's book on 504 past days, must stay ≤ 3% of NAV.
- **Stress tests:** four stress scenarios (2020 crash, 2007 quant unwind, 2009 momentum crash, 2022 rate shock) must each lose ≤ 8% of NAV.
- **Cost stress:** the backtest is re-run at 3× costs.

## Calibration and Status

We simulated 8 regimes: calm, mixed, high-vol, crash, slow alpha decay, regime shift, quant unwind and whipsaw. We tested 11,520 combinations of the de-risking settings and kept the one with the best average rank of return ÷ 95th-percentile drawdown across all regimes. These rules cut worst-case drawdown by about a third in every regime (realistic mix: 22% → 14%), at a cost of about 1–4% annual return.

**Open items:**
1. Re-tune the [sim] values on real out-of-sample predictions.
2. Calibrate the stress shocks against real factor data.
3. Confirm whether the edge falls in high volatility; if it doesn't, the 1.5×/2.5× rule loosens to 2×/3×.
