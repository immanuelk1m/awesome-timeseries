# Awesome Time Series Forecast

Languages: **English** | [한국어](README.ko.md)

> A curated collection of frameworks, agents, libraries, papers, benchmarks, and learning resources for time series forecasting and analysis.

This repository reorganizes useful links for practitioners, competitors, and researchers into a cleaner English-first README, while keeping a separate Korean version.

## Before You Start...

Before you kick off a serious forecasting project, read [Forecast evaluation for data scientists: common pitfalls and best practices](https://pmc.ncbi.nlm.nih.gov/articles/PMC9718476/).

It is one of the most practical papers to read first because it highlights where forecasting projects usually fail before model choice even matters: **data leakage**, **bad validation design**, **misleading error metrics**, and **ignoring strong naive / seasonal naive baselines**. If you have enough historical data, prefer **rolling-origin evaluation / time-series cross-validation**, choose metrics that match your business objective, and beat a solid baseline before reaching for a more complex model.

## Table of Contents
- [Books](#books)
- [Courses](#courses)
- [Papers](#papers)
- [Tutorials and talks](#tutorials-and-talks)

## Books

| Resource | Description | Type |
| --- | --- | --- |
| [Forecasting: Principles and Practice (3rd ed.)](https://otexts.com/fpp3/intro.html) | Free online book covering forecasting fundamentals, statistical methods, and hands-on R workflows. | Book |
| [Machine Learning Systems (Harvard)](https://www.mlsysbook.ai/) | A strong systems-oriented book for deploying and operating ML pipelines, useful when forecasting moves beyond notebooks. | Book |
| [O'Reilly Learning Platform](https://learning.oreilly.com/home/) | Useful if you have institutional access and want a broad catalog of books on ML, statistics, data engineering, and forecasting. | Learning library |

## Courses

| Resource | Description | Format |
| --- | --- | --- |
| [Time Series Forecasting](https://www.youtube.com/watch?v=uwKiT1o1TkI&list=PLyCNZ_xXGzpm7W9jLqbIyBAiSO5jDwJeE&index=1) | Practical step-by-step course playlist focused on time series forecasting workflows. | Video course |
| [Microprediction](https://github.com/microprediction) | Challenge-driven environment for practicing real-time forecasting and probabilistic prediction. | Competition / hands-on learning |
| [O'Reilly Learning Platform](https://learning.oreilly.com/home/) | Good for structured video courses and technical series if your EDU or company account includes access. | Course platform |

## Papers

### Forecasting models

| Model | Description | Paper | Code | Notes |
| --- | --- | --- | --- | --- |
| Autoformer | Uses Auto-Correlation and progressive decomposition for strong long-horizon forecasting. | [arXiv:2106.13008](https://arxiv.org/abs/2106.13008) | [thuml/Autoformer](https://github.com/thuml/Autoformer) | Transformer |
| FEDformer | Frequency-enhanced decomposition with Fourier/Wavelet blocks for efficient long-term forecasting. | [ICML 2022](https://proceedings.mlr.press/v162/zhou22g/zhou22g.pdf) | [MAZiqing/FEDformer](https://github.com/MAZiqing/FEDformer) | Transformer |
| Informer | Introduces ProbSparse attention for scalable long-sequence forecasting. | [arXiv:2012.07436](https://arxiv.org/abs/2012.07436) | [zhouhaoyi/Informer2020](https://github.com/zhouhaoyi/Informer2020) | Transformer |
| iTransformer | Treats variables as tokens to model multivariate dependencies more directly. | [arXiv:2310.06625](https://arxiv.org/abs/2310.06625) | [thuml/iTransformer](https://github.com/thuml/iTransformer) | Transformer |
| Nonstationary Transformer | Stabilizes non-stationary series with normalization and De-stationary Attention. | [arXiv:2205.14415](https://arxiv.org/abs/2205.14415) | [thuml/Nonstationary_Transformers](https://github.com/thuml/Nonstationary_Transformers) | Transformer |
| Temporal Fusion Transformer (TFT) | Attention-based multi-horizon forecasting model with strong interpretability. | [arXiv:1912.09363](https://arxiv.org/abs/1912.09363) | [google-research/tft](https://github.com/google-research/google-research/tree/master/tft) | Transformer |
| TimeXer | Separates endogenous and exogenous representations and links them with global tokens. | [arXiv:2403.09898](https://arxiv.org/pdf/2403.09898) | [thuml/TimeXer](https://github.com/thuml/TimeXer) | Transformer / exogenous variables |
| Probabilistic Carbon Price Transformer | Combines mixed-frequency information with Transformer forecasting for carbon-price quantile prediction. | [Paper](https://www.sciencedirect.com/science/article/pii/S0306261925006816) | - | Probabilistic forecasting |
| Parallel Transformer-CNN | Hybrid model that combines Transformer and CNN branches after secondary decomposition. | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417425035833) | - | Transformer + CNN |
| Temporal-Variable Fusion Transformer | Fuses temporal dependence and variable correlation for electricity-price forecasting. | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S036054422504441X) | - | Energy forecasting |
| HE-DTimeXer | TimeXer-style model for short-term electricity price forecasting with exogenous variables. | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0378779625011277) | - | Energy forecasting |
| TimesNet | Projects 1D time series into 2D representations to learn richer temporal patterns. | [arXiv:2210.02186](https://arxiv.org/abs/2210.02186) | [thuml/TimesNet](https://github.com/thuml/TimesNet) | General-purpose TS model |
| CMamba | Convolutional selective SSM for improved long-range multivariate time series modeling. | [arXiv:2406.05316](https://arxiv.org/pdf/2406.05316) | [zclzcl0223/CMamba](https://github.com/zclzcl0223/CMamba) | Mamba / SSM |
| Mamba | Selective state space model with linear-time sequence modeling. | [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) | [state-spaces/mamba](https://github.com/state-spaces/mamba) | Mamba / SSM |
| S-Mamba | Sparse-gated multi-state-space variant tailored for time series. | [arXiv:2403.11144](https://arxiv.org/abs/2403.11144v3) | [wzhwzhwzh0921/S-D-Mamba](https://github.com/wzhwzhwzh0921/S-D-Mamba) | Mamba / SSM |
| xLSTM-Mixer | Combines xLSTM blocks and Mixer-style token mixing for long-sequence forecasting. | [arXiv:2410.16928](https://arxiv.org/abs/2410.16928) | [mauricekraus/xlstm-mixer](https://github.com/mauricekraus/xlstm-mixer) | RNN + Mixer |
| N-BEATS | Deep residual MLP architecture with backward/forward forecasting blocks. | [arXiv:1905.10437](https://arxiv.org/abs/1905.10437) | [ServiceNow/N-BEATS](https://github.com/ServiceNow/N-BEATS) | MLP |
| TimeMixer | MLP-style architecture with past/future mixing blocks for long- and short-term forecasting. | [OpenReview](https://openreview.net/pdf?id=7oLshfEIC2) | [kwuking/TimeMixer](https://github.com/kwuking/TimeMixer) | MLP Mixer |
| DUET | Dual clustering over time and variables for better long-horizon multivariate forecasting. | [arXiv:2412.10859](https://arxiv.org/abs/2412.10859) | [decisionintelligence/DUET](https://github.com/decisionintelligence/DUET) | Clustering |
| DLinear | Strong decomposition + linear baseline from the LTSF-Linear family. | [arXiv:2205.13504](https://arxiv.org/abs/2205.13504) | [vivva/DLinear](https://github.com/vivva/DLinear) | Linear baseline |
| DeepEDM | Learns time-series dynamics explicitly through a DeepEDM formulation. | [Project Page](https://abrarmajeedi.github.io/deep_edm/) | - | Dynamical systems |

### Time-series anomaly detection

| Model | Description | Paper | Code | Notes |
| --- | --- | --- | --- | --- |
| Telemanom | LSTM + nonparametric dynamic thresholds for telemetry anomaly detection. | [arXiv:1802.04431](https://arxiv.org/abs/1802.04431) | [khundman/telemanom](https://github.com/khundman/telemanom) | LSTM |
| TranAD | Self-conditioning Transformer with adversarial learning for multivariate anomaly detection. | [VLDB](https://vldb.org/pvldb/vol15/p1201-tuli.pdf) | [imperial-qore/TranAD](https://github.com/imperial-qore/TranAD) | Transformer |
| CATCH | Frequency patching and channel-aware modeling for multivariate anomaly detection. | [arXiv:2410.12261](https://arxiv.org/abs/2410.12261) | [decisionintelligence/catch](https://github.com/decisionintelligence/catch) | Frequency domain |
| DualTF | Joint time/frequency-domain anomaly detection with overlapping windows. | [ACM DL](https://dl.acm.org/doi/10.1145/3589334.3645556) | [kaist-dmlab/DualTF](https://github.com/kaist-dmlab/DualTF) | Dual-domain |
| TFMAE | Masked autoencoder with temporal and frequency masking for robust anomaly detection. | [Paper](https://github.com/LMissher/TFMAE/blob/main/paper/TFMAE.pdf) | [LMissher/TFMAE](https://github.com/LMissher/TFMAE) | Masked autoencoder |
| NPSR | Dual reconstruction with nominality scores for point and contextual anomalies. | [arXiv:2310.15416](https://arxiv.org/abs/2310.15416) | [andrewlai61616/NPSR](https://github.com/andrewlai61616/NPSR) | Performer |
| DCdetector | Dual-branch attention and contrastive learning for anomaly representation learning. | [arXiv:2306.10347](https://arxiv.org/abs/2306.10347) | [DAMO-DI-ML/KDD2023-DCdetector](https://github.com/DAMO-DI-ML/KDD2023-DCdetector) | Contrastive learning |
| ModernTCN | Pure convolutional architecture that also supports forecasting, classification, and anomaly detection. | [OpenReview](https://openreview.net/pdf?id=vpJMJerXHU) | [luodhhh/ModernTCN](https://github.com/luodhhh/ModernTCN) | CNN |
| CARLA | Contrastive self-supervised anomaly representation learning without labels. | [Pattern Recognition](https://www.sciencedirect.com/science/article/pii/S0031320324006253) | [zamanzadeh/CARLA](https://github.com/zamanzadeh/CARLA) | Contrastive learning |
| PatchAD | Lightweight multi-scale patch-based MLP-Mixer for anomaly detection. | [arXiv:2401.09793](https://arxiv.org/abs/2401.09793) | [EmorZz1G/PatchAD](https://github.com/EmorZz1G/PatchAD) | MLP-Mixer |

### Foundation models

| Model | Description | Paper | Code | Notes |
| --- | --- | --- | --- | --- |
| Chronos Forecasting | Amazon Science pretrained forecasting family for zero-shot, patch-based, and covariate-aware inference. | [arXiv:2403.07815](https://arxiv.org/abs/2403.07815) | [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) | Pretrained suite |
| Chronos-2 | Extends Chronos toward universal forecasting with grouped attention and in-context learning. | [arXiv:2510.15821](https://arxiv.org/abs/2510.15821) | - | Universal ICL |
| Time-MoE | Billion-scale sparse Mixture-of-Experts time series foundation model trained on Time-300B. | [arXiv:2409.16040](https://arxiv.org/abs/2409.16040) | - | MoE foundation model |
| Sundial | Native time-series foundation model for continuous distribution forecasting. | [arXiv:2502.00816](https://arxiv.org/abs/2502.00816) | - | Distribution forecasting |
| TiRex | xLSTM + in-context learning approach for forecasting with strong state tracking. | [arXiv:2505.23719](https://arxiv.org/abs/2505.23719) | - | xLSTM / ICL |
| Toto | Decoder-only observability-focused foundation model trained on large-scale time series. | [arXiv:2505.14766](https://arxiv.org/abs/2505.14766) | [DataDog/toto](https://github.com/DataDog/toto) | Observability suite |
| Lag-Llama | Open-source probabilistic foundation model with zero-shot and fine-tuning support. | [arXiv:2310.08278](https://arxiv.org/abs/2310.08278) | [time-series-foundation-models/lag-llama](https://github.com/time-series-foundation-models/lag-llama) | Probabilistic |
| Uni2TS / Moirai | Universal Time Series Transformer ecosystem including Moirai and Moirai MoE. | [arXiv:2402.02592](https://arxiv.org/abs/2402.02592) | [SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts) | See [Salesforce Moirai Blog](https://www.salesforce.com/blog/moirai) |
| Moirai 2.0 | Decoder-only follow-up with multi-token prediction for faster and more precise quantile forecasts. | [arXiv:2511.11698](https://arxiv.org/abs/2511.11698) | - | Quantile forecasting |
| TimesFM 2.5 200M | General-purpose foundation Transformer for multi-domain forecasting. | [arXiv:2310.10688](https://arxiv.org/abs/2310.10688) | [google-research/timesfm](https://github.com/google-research/timesfm) | Foundation Transformer |
| TempoPFN | Parallelizable Linear-RNN-style foundation model for univariate forecasting. | [arXiv:2510.25502](https://arxiv.org/pdf/2510.25502) | [automl/TempoPFN](https://github.com/automl/TempoPFN) | Foundation RNN |

### Surveys and selected papers

| Title | Description | Link |
| --- | --- | --- |
| Foundation Models for Time Series Analysis: A Tutorial and Survey | Tutorial survey that organizes time-series foundation models by architecture, pretraining strategy, adaptation method, and modality. | [arXiv:2403.14735](https://arxiv.org/abs/2403.14735) |
| Deep Learning for Time Series Forecasting: A Survey | Broad survey of deep learning forecasting architectures, feature extraction methods, and datasets. | [Springer](https://link.springer.com/article/10.1007/s13042-025-02560-w) |
| Dual-Forecaster: Integrating Textual and Numerical Data for Time Series Forecasting | Multimodal forecasting approach that aligns textual and numerical representations in a shared latent space. | - |
| ChronoSteer: Steerable Time Series Forecasting via Instruction Tuning | Uses textual instructions to steer numerical forecasting behavior. | - |
| Quo Vadis, Unsupervised Time Series Anomaly Detection? | Critiques evaluation protocols in unsupervised anomaly detection and demonstrates the power of simpler baselines. | [GitHub](https://github.com/ssarfraz/QuoVadisTAD) |
| arXiv:2510.02729 | Recent time-series preprint; see the paper for details. | [arXiv:2510.02729](https://arxiv.org/pdf/2510.02729) |

### Datasets, benchmarks, and agent-style research systems

| Resource | Description | Link |
| --- | --- | --- |
| FinMultiTime | 112.6 GB financial multimodal dataset aligning news, tables, chart images, and price time series along the time axis. | Repo planned via [microsoft/TableProvider](https://github.com/microsoft/TableProvider) |
| DeepAnalyze | Research agent designed to autonomously handle data-centric workflows from EDA to reporting. | [ruc-datalab/DeepAnalyze](https://github.com/ruc-datalab/DeepAnalyze) |
| TimeCopilot | LLM-based forecasting/anomaly-detection agent built around Chronos, Moirai, TimesFM, and TimeGPT backends. | [AzulGarza/timecopilot](https://github.com/AzulGarza/timecopilot) |
| Forecasting Experts' Verdict (FEV) | AutoGluon leaderboard for comparing forecasting performance across datasets. | [Hugging Face Space](https://huggingface.co/spaces/autogluon/fev-leaderboard) |
| GIFT Evaluation Leaderboard | Salesforce benchmark leaderboard and evaluation pipeline for global forecasting. | [Hugging Face Space](https://huggingface.co/spaces/Salesforce/GIFT-Eval) |
| TAB | Time-series anomaly detection leaderboard related to CATCH. | [GitHub](https://github.com/decisionintelligence/TAB) |

## Tutorials and talks

### Tutorials, libraries, and frameworks

| Resource | Description | Link |
| --- | --- | --- |
| [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library) | Reproducible experiment framework spanning forecasting, imputation, classification, and anomaly detection. |
| [PyTorch Forecasting](https://github.com/sktime/pytorch-forecasting) | PyTorch Lightning-based library that integrates multiple deep forecasting models. |
| [Kats](https://github.com/facebookresearch/Kats) | Lightweight framework for forecasting, anomaly detection, and feature extraction. |
| [Orion](https://github.com/sintel-dev/Orion) | Unsupervised machine learning pipeline library for time-series anomaly detection. |
| [Alibi Detect](https://github.com/SeldonIO/alibi-detect) | Toolkit for anomaly, outlier, drift, and adversarial detection across modalities including time series. |
| [River](https://github.com/online-ml/river) | Online machine learning library for streaming data and concept drift scenarios. |
| [Darts](https://github.com/unit8co/darts) | Unified forecasting/anomaly-detection interface with both statistical and deep models. |
| [Prophet](https://github.com/facebook/prophet) | Widely used business forecasting library with trend, seasonality, and holiday components. |
| [data-science-template](https://github.com/CodeCutTech/data-science-template) | Reproducible project template for data-science and forecasting workflows. |

### Discussions and reference lists

| Resource | Description | Link |
| --- | --- | --- |
| Why Mamba did not catch on? | Community discussion on the adoption limits of Mamba-style sequence models. | [Reddit Thread](https://www.reddit.com/r/MachineLearning/comments/1hpg91o/d_why_mamba_did_not_catch_on/) |
| Most Time Series Anomaly Detection results are meaningless | Discussion around evaluation quality and meaning in anomaly detection research. | [Reddit Thread](https://www.reddit.com/r/MachineLearning/comments/1gmwxnr/r_most_time_series_anomaly_detection_results_are/) |
| awesome-time-series | Broad awesome-list for packages, papers, and learning resources across time-series topics. | [GitHub](https://github.com/lmmentel/awesome-time-series) |
| awesome-time-series (cuge1995) | Curated list focused on papers, benchmarks, and current research trends. | [GitHub](https://github.com/cuge1995/awesome-time-series) |
| awesome-industrial-anomaly-detection | Curated list of industrial anomaly detection papers, datasets, and methods. | [GitHub](https://github.com/M-3LAB/awesome-industrial-anomaly-detection) |
| ts-anomaly-benchmark | Benchmark repository covering datasets, methods, and metrics for deep anomaly detection. | [GitHub](https://github.com/zamanzadeh/ts-anomaly-benchmark) |
| awesome-TS-anomaly-detection | Comprehensive catalog of time-series anomaly detection tools and datasets. | [GitHub](https://github.com/rob-med/awesome-TS-anomaly-detection) |
| Awesome Multivariate TS Anomaly Detection | Reading list for multivariate anomaly detection papers by year and venue. | [GitHub](https://github.com/lzz19980125/awesome-multivariate-time-series-anomaly-detection-algorithms) |
