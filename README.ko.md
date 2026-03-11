# Awesome Time Series Forecast

언어: [English](README.md) | **한국어**

> 시계열 예측·분석을 위한 프레임워크, 에이전트, 라이브러리, 논문, 벤치마크, 학습 자료 큐레이션

실무자, 대회 참가자, 연구자가 빠르게 출발할 수 있도록 자료를 다시 정리한 목록. 영문 메인 README와 한국어 별도 README 제공.

## 시작하기 전에

본격적인 시계열 예측 프로젝트 시작 전, [Forecast evaluation for data scientists: common pitfalls and best practices](https://pmc.ncbi.nlm.nih.gov/articles/PMC9718476/) 선독 권장.

모델 구조보다 앞서 예측 프로젝트를 망치기 쉬운 지점을 실용적으로 정리한 논문. 특히 **데이터 누수**, **잘못된 검증 설계**, **부적절한 에러 지표**, **naive / seasonal naive 같은 강한 베이스라인 무시**의 위험성 강조. 데이터가 충분하다면 **rolling-origin evaluation / tsCV** 우선 고려, 업무 목적에 맞는 지표 선택, 복잡한 모델보다 강한 베이스라인 선검증 권장.

## Table of Contents
- [Books](#books)
- [Courses](#courses)
- [Papers](#papers)
- [Tutorials and talks](#tutorials-and-talks)

## Books

| 자료 | 설명 | 유형 |
| --- | --- | --- |
| [Forecasting: Principles and Practice (3rd ed.)](https://otexts.com/fpp3/intro.html) | 예측 기초, 통계적 방법, R 실습을 폭넓게 다루는 무료 온라인 교재 | Book |
| [Machine Learning Systems (Harvard)](https://www.mlsysbook.ai/) | 예측 모델을 노트북 밖 운영 환경으로 옮길 때 도움이 되는 시스템 관점 교재 | Book |
| [O'Reilly Learning Platform](https://learning.oreilly.com/home/) | 기관 계정이 있다면 ML, 통계, 데이터 엔지니어링, 예측 관련 전자책을 폭넓게 볼 수 있는 학습 플랫폼 | Learning library |

## Courses

| 자료 | 설명 | 형식 |
| --- | --- | --- |
| [Forecasting: Principles & Practice 3rd ed](https://www.youtube.com/watch?v=uwKiT1o1TkI&list=PLyCNZ_xXGzpm7W9jLqbIyBAiSO5jDwJeE&index=1) | FPP3 예측 교재를 보완하는 비디오 플레이리스트, 핵심 예측 개념과 방법 중심 | Video course |
| [Microprediction](https://github.com/microprediction) | prediction stream, forecasting, 관련 책/프로젝트를 포함한 Microprediction 생태계 진입용 리소스 허브 | Resource hub |
| [O'Reilly Learning Platform](https://learning.oreilly.com/home/) | EDU 또는 회사 계정 기반의 구조화된 비디오 강의·기술 시리즈 학습 플랫폼 | Course platform |

## Papers

### Forecasting models

| 모델 | 설명 | 논문 | 코드 | 비고 |
| --- | --- | --- | --- | --- |
| Autoformer | Auto-Correlation과 점진적 분해 기반 장기 예측 모델 | [arXiv:2106.13008](https://arxiv.org/abs/2106.13008) | [thuml/Autoformer](https://github.com/thuml/Autoformer) | Transformer |
| FEDformer | Fourier/Wavelet 기반 분해를 사용하는 장기 예측 효율화 모델 | [ICML 2022](https://proceedings.mlr.press/v162/zhou22g/zhou22g.pdf) | [MAZiqing/FEDformer](https://github.com/MAZiqing/FEDformer) | Transformer |
| Informer | ProbSparse attention 기반 장시퀀스 예측 구조 | [arXiv:2012.07436](https://arxiv.org/abs/2012.07436) | [zhouhaoyi/Informer2020](https://github.com/zhouhaoyi/Informer2020) | Transformer |
| iTransformer | 변수를 토큰처럼 다뤄 다변량 의존성을 직접 학습하는 구조 | [arXiv:2310.06625](https://arxiv.org/abs/2310.06625) | [thuml/iTransformer](https://github.com/thuml/iTransformer) | Transformer |
| Nonstationary Transformer | 정규화와 De-stationary Attention 기반 비정상 시계열 대응 모델 | [arXiv:2205.14415](https://arxiv.org/abs/2205.14415) | [thuml/Nonstationary_Transformers](https://github.com/thuml/Nonstationary_Transformers) | Transformer |
| Temporal Fusion Transformer (TFT) | 다중 지평선 예측과 변수 중요도 해석에 강한 attention 기반 모델 | [arXiv:1912.09363](https://arxiv.org/abs/1912.09363) | [google-research/tft](https://github.com/google-research/google-research/tree/master/tft) | Transformer |
| TimeXer | endogenous / exogenous 표현 분리와 global token 연결 구조 | [arXiv:2403.09898](https://arxiv.org/pdf/2403.09898) | [thuml/TimeXer](https://github.com/thuml/TimeXer) | Transformer / exogenous variables |
| Probabilistic Carbon Price Transformer | quantile regression, Transformer, mixed-frequency MIDAS modeling을 결합한 QRTransformer-MIDAS 기반 탄소가격 점·구간·확률 예측 모델 | [Paper](https://www.sciencedirect.com/science/article/pii/S0306261925006816) | - | Probabilistic forecasting / mixed-frequency |
| Parallel Transformer-CNN | 2차 분해 이후 Transformer와 CNN 분기를 결합한 하이브리드 예측 구조 | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417425035833) | - | Transformer + CNN |
| Temporal-Variable Fusion Transformer | 시간 의존성과 변수 상관관계를 함께 융합한 전력가격 예측 모델 | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S036054422504441X) | - | Energy forecasting |
| HE-DTimeXer | 외생 변수를 활용하는 단기 전력가격 예측용 TimeXer 계열 모델 | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0378779625011277) | - | Energy forecasting |
| TimesNet | 1D 시계열을 2D 표현으로 투영해 더 풍부한 패턴을 학습하는 모델 | [arXiv:2210.02186](https://arxiv.org/abs/2210.02186) | [thuml/TimesNet](https://github.com/thuml/TimesNet) | General-purpose TS model |
| CMamba | 다변량 시계열 예측용 channel-correlation-enhanced state space model | [arXiv:2406.05316](https://arxiv.org/abs/2406.05316) | [zclzcl0223/CMamba](https://github.com/zclzcl0223/CMamba) | Mamba / SSM |
| Mamba | 선형 시간 복잡도로 긴 시퀀스를 다루는 selective state space model | [arXiv:2312.00752](https://arxiv.org/abs/2312.00752) | [state-spaces/mamba](https://github.com/state-spaces/mamba) | Mamba / SSM |
| S-Mamba | bidirectional Mamba block과 feed-forward network 기반 변수 상관·시간 의존성 동시 모델링 구조 | [arXiv:2403.11144](https://arxiv.org/abs/2403.11144) | [wzhwzhwzh0921/S-D-Mamba](https://github.com/wzhwzhwzh0921/S-D-Mamba) | Mamba / SSM |
| xLSTM-Mixer | xLSTM 블록과 Mixer 스타일 혼합을 결합한 장기 시퀀스 예측 모델 | [arXiv:2410.16928](https://arxiv.org/abs/2410.16928) | [mauricekraus/xlstm-mixer](https://github.com/mauricekraus/xlstm-mixer) | RNN + Mixer |
| N-BEATS | backward / forward 블록 기반 대표적 deep residual MLP 예측 모델 | [arXiv:1905.10437](https://arxiv.org/abs/1905.10437) | [ServiceNow/N-BEATS](https://github.com/ServiceNow/N-BEATS) | MLP |
| TimeMixer | past / future mixing 블록 기반 MLP 스타일 예측 아키텍처 | [OpenReview](https://openreview.net/pdf?id=7oLshfEIC2) | [kwuking/TimeMixer](https://github.com/kwuking/TimeMixer) | MLP Mixer |
| DUET | 시간축과 변수축 이중 클러스터링 기반 다변량 장기 예측 모델 | [arXiv:2412.10859](https://arxiv.org/abs/2412.10859) | [decisionintelligence/DUET](https://github.com/decisionintelligence/DUET) | Clustering |
| DLinear | LTSF-Linear 계열의 강력한 분해 + 선형 베이스라인 | [arXiv:2205.13504](https://arxiv.org/abs/2205.13504) | [vivva/DLinear](https://github.com/vivva/DLinear) | Linear baseline |
| DeepEDM | DeepEDM 방식의 시계열 동역학 명시적 학습 접근 | [Project Page](https://abrarmajeedi.github.io/deep_edm/) | - | Dynamical systems |

### Time-series anomaly detection

| 모델 | 설명 | 논문 | 코드 | 비고 |
| --- | --- | --- | --- | --- |
| Telemanom | LSTM과 비모수 동적 임계값 기반 텔레메트리 이상 탐지 모델 | [arXiv:1802.04431](https://arxiv.org/abs/1802.04431) | [khundman/telemanom](https://github.com/khundman/telemanom) | LSTM |
| TranAD | self-conditioning Transformer와 adversarial learning 기반 다변량 이상 탐지 모델 | [VLDB](https://vldb.org/pvldb/vol15/p1201-tuli.pdf) | [imperial-qore/TranAD](https://github.com/imperial-qore/TranAD) | Transformer |
| CATCH | 주파수 패칭과 채널 인식 구조 기반 다변량 이상 탐지 모델 | [arXiv:2410.12261](https://arxiv.org/abs/2410.12261) | [decisionintelligence/catch](https://github.com/decisionintelligence/catch) | Frequency domain |
| DualTF | 시간/주파수 이중 도메인 기반 이상 탐지 모델 | [ACM DL](https://dl.acm.org/doi/10.1145/3589334.3645556) | [kaist-dmlab/DualTF](https://github.com/kaist-dmlab/DualTF) | Dual-domain |
| TFMAE | 시간·주파수 마스킹 기반 masked autoencoder 이상 탐지 모델 | [Paper](https://github.com/LMissher/TFMAE/blob/main/paper/TFMAE.pdf) | [LMissher/TFMAE](https://github.com/LMissher/TFMAE) | Masked autoencoder |
| NPSR | nominality score 기반 이중 재구성형 point/contextual anomaly 탐지 모델 | [arXiv:2310.15416](https://arxiv.org/abs/2310.15416) | [andrewlai61616/NPSR](https://github.com/andrewlai61616/NPSR) | Performer |
| DCdetector | dual-branch attention과 contrastive learning 기반 이상 표현 학습 모델 | [arXiv:2306.10347](https://arxiv.org/abs/2306.10347) | [DAMO-DI-ML/KDD2023-DCdetector](https://github.com/DAMO-DI-ML/KDD2023-DCdetector) | Contrastive learning |
| ModernTCN | 예측, 분류, 이상 탐지까지 지원하는 순수 convolutional 구조 | [OpenReview](https://openreview.net/pdf?id=vpJMJerXHU) | [luodhhh/ModernTCN](https://github.com/luodhhh/ModernTCN) | CNN |
| CARLA | 레이블 없이 contrastive self-supervised 방식으로 이상 표현을 학습하는 모델 | [Pattern Recognition](https://www.sciencedirect.com/science/article/pii/S0031320324006253) | [zamanzadeh/CARLA](https://github.com/zamanzadeh/CARLA) | Contrastive learning |
| PatchAD | 경량 multi-scale patch 기반 MLP-Mixer 이상 탐지 모델 | [arXiv:2401.09793](https://arxiv.org/abs/2401.09793) | [EmorZz1G/PatchAD](https://github.com/EmorZz1G/PatchAD) | MLP-Mixer |

### Foundation models

| 모델 | 설명 | 논문 | 코드 | 비고 |
| --- | --- | --- | --- | --- |
| Chronos Forecasting | zero-shot, patch 기반, covariate-aware 추론을 지원하는 Amazon Science의 사전학습 예측 모델 군 | [arXiv:2403.07815](https://arxiv.org/abs/2403.07815) | [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) | Pretrained suite |
| Chronos-2 | grouped attention과 in-context learning을 통해 universal forecasting 방향으로 확장된 후속 모델 | [arXiv:2510.15821](https://arxiv.org/abs/2510.15821) | - | Universal ICL |
| Time-MoE | Time-300B로 학습한 billion-scale sparse Mixture-of-Experts 기반 시계열 파운데이션 모델 | [arXiv:2409.16040](https://arxiv.org/abs/2409.16040) | - | MoE foundation model |
| Sundial | 연속 분포 예측에 초점을 둔 native time-series foundation model | [arXiv:2502.00816](https://arxiv.org/abs/2502.00816) | - | Distribution forecasting |
| TiRex | short/long horizon 모두를 겨냥한 xLSTM 기반 zero-shot forecasting 모델, 향상된 in-context learning과 state tracking 강조 | [arXiv:2505.23719](https://arxiv.org/abs/2505.23719) | - | xLSTM / ICL / Zero-shot |
| Toto | 대규모 시계열로 학습된 observability 중심 decoder-only foundation model | [arXiv:2505.14766](https://arxiv.org/abs/2505.14766) | [DataDog/toto](https://github.com/DataDog/toto) | Observability suite |
| Lag-Llama | zero-shot과 fine-tuning을 모두 지원하는 오픈소스 확률 예측 파운데이션 모델 | [arXiv:2310.08278](https://arxiv.org/abs/2310.08278) | [time-series-foundation-models/lag-llama](https://github.com/time-series-foundation-models/lag-llama) | Probabilistic |
| Uni2TS / Moirai | Moirai와 Moirai MoE를 포함하는 Universal Time Series Transformer 생태계 | [arXiv:2402.02592](https://arxiv.org/abs/2402.02592) | [SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts) | See [Salesforce Moirai Blog](https://www.salesforce.com/blog/moirai) |
| Moirai 2.0 | quantile forecasting과 multi-token prediction 기반 decoder-only 시계열 파운데이션 모델, 이전 Moirai 대비 확률 예측 정확도·추론 효율·모델 크기 trade-off 개선 | [arXiv:2511.11698](https://arxiv.org/abs/2511.11698) | - | Decoder-only / Quantile forecasting |
| TimesFM 2.5 200M | 다양한 도메인에 적용 가능한 범용 시계열 파운데이션 Transformer | [arXiv:2310.10688](https://arxiv.org/abs/2310.10688) | [google-research/timesfm](https://github.com/google-research/timesfm) | Foundation Transformer |
| TempoPFN | 병렬화 가능한 Linear-RNN 스타일의 단변량 예측 파운데이션 모델 | [arXiv:2510.25502](https://arxiv.org/pdf/2510.25502) | [automl/TempoPFN](https://github.com/automl/TempoPFN) | Foundation RNN |

### Surveys and selected papers

| 제목 | 설명 | 링크 |
| --- | --- | --- |
| Foundation Models for Time Series Analysis: A Tutorial and Survey | 아키텍처, 사전학습 전략, 적응 방식, 모달리티 기준으로 foundation model을 정리한 튜토리얼 서베이 | [arXiv:2403.14735](https://arxiv.org/abs/2403.14735) |
| Deep Learning for Time Series Forecasting: A Survey | 딥러닝 기반 예측 구조, 특징 추출 방식, 데이터셋을 폭넓게 다루는 서베이 | [Springer](https://link.springer.com/article/10.1007/s13042-025-02560-w) |
| Dual-Forecaster: Integrating Textual and Numerical Data for Time Series Forecasting | 1차 출처 논문/저장소 링크 확인 전까지 세부 방법 설명 보류가 필요한 미검증 항목 | - |
| ChronoSteer: Steerable Time Series Forecasting via Instruction Tuning | 1차 출처 논문/저장소 링크 확인 전까지 세부 방법 설명 보류가 필요한 미검증 항목 | - |
| Quo Vadis, Unsupervised Time Series Anomaly Detection? | 비지도 이상 탐지 평가 프로토콜 비판과 단순 베이스라인 강점 제시 | [GitHub](https://github.com/ssarfraz/QuoVadisTAD) |
| arXiv:2510.02729 | 최신 시계열 프리프린트, 자세한 내용은 원문 참고 | [arXiv:2510.02729](https://arxiv.org/pdf/2510.02729) |

### Datasets, benchmarks, and agent-style research systems

| 자료 | 설명 | 링크 |
| --- | --- | --- |
| FinMultiTime | 미국·중국 시장을 아우르며 뉴스, 금융 테이블, K-line 차트, 주가 시계열을 정렬한 4모달 이중언어 금융 시계열 데이터셋 | [arXiv:2506.05019](https://arxiv.org/abs/2506.05019) |
| DeepAnalyze | EDA부터 리포트까지 데이터 중심 업무를 자율 처리하도록 설계된 연구용 에이전트 | [ruc-datalab/DeepAnalyze](https://github.com/ruc-datalab/DeepAnalyze) |
| TimeCopilot | Chronos, Moirai, TimesFM, TimeGPT 같은 시계열 파운데이션 모델과 LLM을 결합해 예측 워크플로를 자동화·설명하는 오픈소스 forecasting agent | [TimeCopilot/timecopilot](https://github.com/TimeCopilot/timecopilot) |
| Forecasting Experts' Verdict (FEV) | 다양한 데이터셋에서 예측 성능 비교가 가능한 AutoGluon 리더보드 | [Hugging Face Space](https://huggingface.co/spaces/autogluon/fev-leaderboard) |
| GIFT Evaluation Leaderboard | GIFT-Eval 프로젝트의 general time series forecasting 벤치마크 및 리더보드 | [Hugging Face Space](https://huggingface.co/spaces/Salesforce/GIFT-Eval) |
| TAB | 평가 파이프라인, 베이스라인, 메트릭, 공개 결과를 포함하는 시계열 이상 탐지 통합 벤치마킹 라이브러리 겸 리더보드 | [GitHub](https://github.com/decisionintelligence/TAB) |

## Tutorials and talks

### Tutorials, libraries, and frameworks

| 자료 | 설명 | 링크 |
| --- | --- | --- |
| ⭐ Forecasting for Data Scientists (FFDS) | 데이터 사이언티스트를 위한 실전 예측 주제 플레이리스트 | [YouTube Playlist](https://www.youtube.com/playlist?list=PLSpAVARuzDa2KlLPVynFc5uBZ0U_cXF6X) |
| [Time-Series-Library (TSLib)](https://github.com/thuml/Time-Series-Library) | 예측, 결측치 대체, 분류, 이상 탐지까지 아우르는 재현성 높은 실험 프레임워크 |
| [PyTorch Forecasting](https://github.com/sktime/pytorch-forecasting) | 여러 딥러닝 예측 모델을 통합한 PyTorch Lightning 기반 라이브러리 |
| [Kats](https://github.com/facebookresearch/Kats) | 예측, 이상 탐지, 특징 추출을 지원하는 경량 프레임워크 |
| [Orion](https://github.com/sintel-dev/Orion) | 비지도 시계열 이상 탐지용 머신러닝 파이프라인 라이브러리 |
| [Alibi Detect](https://github.com/SeldonIO/alibi-detect) | 시계열을 포함한 다양한 데이터에서 이상치, 드리프트, 적대적 예제를 탐지하는 툴킷 |
| [River](https://github.com/online-ml/river) | 스트리밍 데이터와 concept drift 환경에 강한 온라인 머신러닝 라이브러리 |
| [Darts](https://github.com/unit8co/darts) | 통계 모델과 딥러닝 모델을 함께 제공하는 통합 예측/이상탐지 인터페이스 |
| [Prophet](https://github.com/facebook/prophet) | 추세, 계절성, 휴일 효과를 쉽게 다루는 대표적 비즈니스 예측 라이브러리 |
| [data-science-template](https://github.com/CodeCutTech/data-science-template) | 유지보수 가능하고 재현 가능한 데이터 사이언스 프로젝트용 cookiecutter 스타일 템플릿 |

### Discussions and reference lists

| 자료 | 설명 | 링크 |
| --- | --- | --- |
| Why Mamba did not catch on? | Mamba 계열 시퀀스 모델이 널리 채택되지 않은 이유를 다루는 커뮤니티 토론 | [Reddit Thread](https://www.reddit.com/r/MachineLearning/comments/1hpg91o/d_why_mamba_did_not_catch_on/) |
| Most Time Series Anomaly Detection results are meaningless | 이상 탐지 연구에서 평가 품질과 결과 해석 문제를 다루는 토론 | [Reddit Thread](https://www.reddit.com/r/MachineLearning/comments/1gmwxnr/r_most_time_series_anomaly_detection_results_are/) |
| awesome-time-series | 시계열 전반의 패키지, 논문, 학습 자료를 폭넓게 모은 awesome-list | [GitHub](https://github.com/lmmentel/awesome-time-series) |
| awesome-time-series (cuge1995) | 논문, 벤치마크, 최신 연구 동향 중심 큐레이션 | [GitHub](https://github.com/cuge1995/awesome-time-series) |
| awesome-industrial-anomaly-detection | 산업용 이미지 이상 탐지와 결함 탐지 관련 논문, 데이터셋, 벤치마크 리스트 | [GitHub](https://github.com/M-3LAB/awesome-industrial-anomaly-detection) |
| ts-anomaly-benchmark | 딥러닝 기반 이상 탐지용 데이터셋, 방법론, 메트릭 정리 벤치마크 저장소 | [GitHub](https://github.com/zamanzadeh/ts-anomaly-benchmark) |
| awesome-TS-anomaly-detection | 시계열 이상 탐지 도구와 데이터셋 종합 카탈로그 | [GitHub](https://github.com/rob-med/awesome-TS-anomaly-detection) |
| Awesome Multivariate TS Anomaly Detection | 연도·학회 기준으로 정리된 다변량 이상 탐지 읽기 리스트 | [GitHub](https://github.com/lzz19980125/awesome-multivariate-time-series-anomaly-detection-algorithms) |
