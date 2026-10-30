Simulated orders: 15000 per seed · chronological 80/20 split · seeds [0, 1, 2] · mean ± std

| Model | P50 MAE (min) ↓ | P95 coverage (target 0.95) | P95 pinball loss ↓ |
|---|---|---|---|
| Constant baseline (training-set quantiles) | 9.54 ± 1.04 | 0.945 ± 0.017 | 1.870 ± 0.499 |
| **DeepPrep (Wide & Deep + queue features)** | 5.56 ± 0.68 | 0.927 ± 0.015 | 1.088 ± 0.303 |
| DeepPrep without queue features (ablation) | 5.50 ± 0.45 | 0.922 ± 0.015 | 1.049 ± 0.229 |
| Gradient-boosted quantile regression, same inputs | 4.69 ± 0.51 | 0.915 ± 0.017 | 1.024 ± 0.266 |
| Gradient-boosted quantile regression + item count | 4.63 ± 0.50 | 0.920 ± 0.017 | 1.013 ± 0.274 |
