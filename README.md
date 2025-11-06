# Awesome Time Series Forecast

> 시계열 예측·분석 위한 **프레임워크, 에이전트, 라이브러리, 큐레이션 모음**. (실무 · 대회 · 연구 어디서든 바로 써먹을 링크 제공 목표.)

`awesome-time-series` 레포 정리 스타일 참고해 큐레이션 재구성. 요청 따라 **모델 / 에이전트 / 학습자료 / 리더보드** 체계 사용.

## Contents
- [Models - TS Forecast](#models---ts-forecast) 예측 모델
- [Models - TS Anomaly Detection](#models---ts-anomaly-detection) 이상 탐지 모델
- [Agents](#agents) 항목
- [Papers](#papers) 논문
- [Learning Resources](#learning-resources) 참고용
- [Utilities](#utilities) 도구
- [Community](#community) 토론
- [Related References](#related-references) 큐레이션
- [Leaderboards](#leaderboards) 정리

## Models - TS Forecast

| Model | Description | Paper | GitHub | 비고 |
| --- | --- | --- | --- | --- |
| TimesFM 2.5 200M | 다중 도메인 시계열용 파운데이션 Transformer 모델 | [arXiv:2310.10688](https://arxiv.org/abs/2310.10688) | [google-research/timesfm](https://github.com/google-research/timesfm) | Foundation Transformer |
| Autoformer | Auto-Correlation 기반 자동 분해로 장기 예측 성능 높인 모델 | [arXiv:2106.13008](https://arxiv.org/abs/2106.13008) | [thuml/Autoformer](https://github.com/thuml/Autoformer) | Transformer |
| FEDformer | Fourier/Wavelet 기반 주파수 분해로 장기 예측 효율 높인 모델 | [ICML 2022](https://proceedings.mlr.press/v162/zhou22g/zhou22g.pdf) | [MAZiqing/FEDformer](https://github.com/MAZiqing/FEDformer) | Transformer |
| Informer | ProbSparse Attention으로 긴 시계열 효율적으로 모델링하는 구조 | [arXiv:2012.07436](https://arxiv.org/abs/2012.07436) | [zhouhaoyi/Informer2020](https://github.com/zhouhaoyi/Informer2020) | Transformer |
| iTransformer | 변수를 토큰으로 다루는 Inverted Transformer로 다변량 상관관계 학습 | [arXiv:2310.06625](https://arxiv.org/abs/2310.06625) | [thuml/iTransformer](https://github.com/thuml/iTransformer) | Transformer |
| Nonstationary Transformer | 시계열 정상화와 De-stationary Attention으로 비정상 데이터 안정화 | [arXiv:2205.14415](https://arxiv.org/abs/2205.14415) | [thuml/Nonstationary_Transformers](https://github.com/thuml/Nonstationary_Transformers) | Transformer |
| Temporal Fusion Transformer (TFT) | 다중 지평선 예측에 적합한 주의 기반 아키텍처로 변수 중요도 해석 가능 | [arXiv:1912.09363](https://arxiv.org/abs/1912.09363) | [google-research/tft](https://github.com/google-research/google-research/tree/master/tft) | Transformer |
| TimeXer | 외생 변수 통합해 패치·변수 단위 주의 메커니즘으로 SOTA 기록 | [arXiv:2403.09898](https://arxiv.org/pdf/2403.09898) | [thuml/TimeXer](https://github.com/thuml/TimeXer) | Transformer |
| TimesNet | 1D 시계열을 2D 텐서로 투영해 공간 패턴 학습하는 범용 모델 | [arXiv:2210.02186](https://arxiv.org/abs/2210.02186) | [thuml/TimesNet](https://github.com/thuml/TimesNet) | Transformer-like |
| CMamba | Convolutional Selective SSM으로 다변량 시계열 장기 의존성 개선 | [arXiv:2406.05316](https://arxiv.org/pdf/2406.05316) | [zclzcl0223/CMamba](https://github.com/zclzcl0223/CMamba) | SSM (Mamba) |
| Mamba | Selective State Space 모델로 선형 시간 복잡도로 긴 시퀀스 학습 | [arXiv:2405.21060](https://arxiv.org/pdf/2405.21060) | [state-spaces/mamba](https://github.com/state-spaces/mamba) | SSM (Mamba) |
| S-Mamba | Sparse gating과 다중 상태 공간 결합으로 시계열 전용 Mamba 변형 구성 | [arXiv:2403.11144](https://arxiv.org/abs/2403.11144v3) | [wzhwzhwzh0921/S-D-Mamba](https://github.com/wzhwzhwzh0921/S-D-Mamba) | SSM (Mamba) |
| xLSTM-Mixer | xLSTM 블록과 Mixer 결합해 긴 시퀀스 정보 혼합 최적화 | [arXiv:2410.16928](https://arxiv.org/abs/2410.16928) | [mauricekraus/xlstm-mixer](https://github.com/mauricekraus/xlstm-mixer) | RNN + Mixer |
| N-BEATS | Residual stack과 forward/backward 분해 활용한 MLP 기반 모델 | [arXiv:1905.10437](https://arxiv.org/abs/1905.10437) | [ServiceNow/N-BEATS](https://github.com/ServiceNow/N-BEATS) | MLP |
| TimeMixer | Past/Future Mixing 블록 탑재한 MLP 기반 장·단기 예측 모델(TimeMixer++) | [OpenReview](https://openreview.net/pdf?id=7oLshfEIC2) | [kwuking/TimeMixer](https://github.com/kwuking/TimeMixer) | MLP Mixer |
| DUET | 시간·채널 듀얼 클러스터링으로 다변량 시계열 패턴 학습해 장기 예측 정확도 향상 (KDD 2025) | [arXiv:2412.10859](https://arxiv.org/abs/2412.10859) | [decisionintelligence/DUET](https://github.com/decisionintelligence/DUET) | Clustering |
| DLinear | 분해 + 선형 회귀 활용해 간결하면서 강력한 예측 베이스라인 제공 | [arXiv:2205.13504](https://arxiv.org/abs/2205.13504) | [vivva/DLinear](https://github.com/vivva/DLinear) | Linear |
| DeepEDM | 시계열 동역학 명시적으로 학습하는 DeepEDM 접근법 | [Project Page](https://abrarmajeedi.github.io/deep_edm/) | - | Dynamics |

## Models - TS Anomaly Detection

| Model | Description | Paper | GitHub | 비고 |
| --- | --- | --- | --- | --- |
| Telemanom | LSTM과 비모수 동적 임계값으로 우주선 텔레메트리 이상 탐지 (NASA SMAP/Curiosity 데이터 활용) | [arXiv:1802.04431](https://arxiv.org/abs/1802.04431) | [khundman/telemanom](https://github.com/khundman/telemanom) | LSTM |
| TranAD | Transformer 자기조건화와 adversarial 학습으로 다변량 시계열 이상 탐지 (VLDB 2022) | [VLDB](https://vldb.org/pvldb/vol15/p1201-tuli.pdf) | [imperial-qore/TranAD](https://github.com/imperial-qore/TranAD) | Transformer |
| CATCH | 주파수 패칭과 채널 인식 메커니즘으로 다변량 시계열 이상 탐지 (ICLR 2025) | [arXiv:2410.12261](https://arxiv.org/abs/2410.12261) | [decisionintelligence/catch](https://github.com/decisionintelligence/catch) | Frequency Domain |
| DualTF | 시간·주파수 이중 도메인 중첩 윈도우로 패턴 기반 이상치 탐지 정확도 향상 (TheWebConf 2024) | [ACM DL](https://dl.acm.org/doi/10.1145/3589334.3645556) | [kaist-dmlab/DualTF](https://github.com/kaist-dmlab/DualTF) | Dual-Domain |
| TFMAE | Masked Autoencoder와 시간·주파수 마스킹으로 분포 변화에 강건한 이상 탐지 (ICDE 2024) | [Paper](https://github.com/LMissher/TFMAE/blob/main/paper/TFMAE.pdf) | [LMissher/TFMAE](https://github.com/LMissher/TFMAE) | Masked Autoencoder |
| NPSR | Nominality Score 기반 듀얼 재구성으로 point·contextual 이상치 동시 탐지 (NeurIPS 2023) | [arXiv:2310.15416](https://arxiv.org/abs/2310.15416) | [andrewlai61616/NPSR](https://github.com/andrewlai61616/NPSR) | Performer |
| DCdetector | 듀얼 브랜치 어텐션과 순수 대조 학습으로 이상치 판별 표현 학습 (KDD 2023) | [arXiv:2306.10347](https://arxiv.org/abs/2306.10347) | [DAMO-DI-ML/KDD2023-DCdetector](https://github.com/DAMO-DI-ML/KDD2023-DCdetector) | Contrastive Learning |
| CARLA | 대조 학습 기반 자기지도 학습으로 레이블 없이 시계열 이상치 표현 학습 | [Pattern Recognition](https://www.sciencedirect.com/science/article/pii/S0031320324006253) | [zamanzadeh/CARLA](https://github.com/zamanzadeh/CARLA) | Contrastive Learning |
| PatchAD | 대조 학습과 다중 스케일 패치 기반 경량 MLP-Mixer로 시계열 이상 탐지 | [arXiv:2401.09793](https://arxiv.org/abs/2401.09793) | [EmorZz1G/PatchAD](https://github.com/EmorZz1G/PatchAD) | MLP-Mixer |

## Agents

| Agent | Description | GitHub |
| --- | --- | --- |
| DeepAnalyze | 데이터 중심 업무(EDA→전처리→모델링→리포트)를 에이전트형 LLM이 자율 처리하도록 설계 | [ruc-datalab/DeepAnalyze](https://github.com/ruc-datalab/DeepAnalyze) |
| TimeCopilot | LLM과 Chronos/Moirai/TimesFM/TimeGPT 백엔드 결합해 예측·교차검증·이상탐지 자동화 | [AzulGarza/timecopilot](https://github.com/AzulGarza/timecopilot) |

## Papers

| Title | Description | Link |
| --- | --- | --- |
| arXiv:2510.02729 | 최신 시계열 관련 프리프린트 (세부 내용은 원문 참고) | [arXiv:2510.02729](https://arxiv.org/pdf/2510.02729) |
| Quo Vadis, Unsupervised Time Series Anomaly Detection? | 비지도 시계열 이상 탐지 연구의 평가 메트릭·벤치마킹 문제 분석하고 단순한 베이스라인 효과 입증 (ICML 2024 Position Paper) | [GitHub](https://github.com/ssarfraz/QuoVadisTAD) |
| Awesome Multivariate TS Anomaly Detection | 다변량 시계열 이상 탐지 논문을 연도·학회별로 정리한 읽기 리스트 | [GitHub](https://github.com/lzz19980125/awesome-multivariate-time-series-anomaly-detection-algorithms) |

## Learning Resources

| Resource | Description | Link Type |
| --- | --- | --- |
| Time Series Forecasting | 시계열 예측 단계별로 다루는 실습 중심 강의 시리즈 | [YouTube](https://www.youtube.com/watch?v=uwKiT1o1TkI&list=PLyCNZ_xXGzpm7W9jLqbIyBAiSO5jDwJeE&index=1) |
| Forecasting: Principles and Practice (3rd ed.) | 통계 기반 시계열 분석 이론과 R 실습 제공하는 무료 온라인 교재 | [Book](https://otexts.com/fpp3/intro.html) |
| Microprediction | 실시간 예측 경진대회와 오픈소스 생태계로 다양한 챌린지 경험 가능 | [GitHub](https://github.com/microprediction) |
| Time-Series-Library (TSLib) | 예측·대체·분류·이상탐지 등 5대 작업 재현성 있게 지원하는 실험 프레임워크 | [GitHub](https://github.com/thuml/Time-Series-Library) |
| Machine Learning Systems (Harvard) | 하버드대가 무료로 공개한 머신러닝 시스템 설계·운영 교재 | [Book](https://www.mlsysbook.ai/) |
| O'Reilly Learning Platform | EDU 계정 있으면 다양한 머신러닝·데이터 과학 전자책 무료로 열람 가능 | [Learning Portal](https://learning.oreilly.com/home/) |

## Utilities

| Utility | Description | GitHub |
| --- | --- | --- |
| PyTorch Forecasting | PyTorch Lightning 기반 시계열 예측 라이브러리로 다양한 딥러닝 모델 통합 제공 | [sktime/pytorch-forecasting](https://github.com/sktime/pytorch-forecasting) |
| data-science-template | 데이터 사이언스 프로젝트를 표준 구조로 정의해 재현성과 유지보수성을 높이는 템플릿 | [CodeCutTech/data-science-template](https://github.com/CodeCutTech/data-science-template) |
| Kats | 시계열 분석·예측·이상 탐지·특징 추출을 경량 확장 가능한 프레임워크로 제공 (Facebook Research) | [facebookresearch/Kats](https://github.com/facebookresearch/Kats) |
| Orion | 비지도 학습 기반 시계열 이상 탐지용 머신러닝 파이프라인 라이브러리 | [sintel-dev/Orion](https://github.com/sintel-dev/Orion) |
| Alibi Detect | 시계열·이미지·텍스트 등 다양한 데이터에서 이상치·드리프트·적대적 예제 탐지 라이브러리 | [SeldonIO/alibi-detect](https://github.com/SeldonIO/alibi-detect) |
| River | 스트리밍 데이터 온라인 학습용 머신러닝 라이브러리로 Concept Drift 대응 가능 | [online-ml/river](https://github.com/online-ml/river) |
| Darts | 통계 모델과 딥러닝 방법을 scikit-learn 스타일 통합 인터페이스로 제공하는 시계열 예측·이상 탐지 라이브러리 | [unit8co/darts](https://github.com/unit8co/darts) |

## Community

| Topic | Description | Link |
| --- | --- | --- |
| Why Mamba did not catch on? | Mamba 계열 모델 채택과 한계에 대한 커뮤니티 토론 | [Reddit Thread](https://www.reddit.com/r/MachineLearning/comments/1hpg91o/d_why_mamba_did_not_catch_on/) |

## Related References

| Repository | Description | Link |
| --- | --- | --- |
| awesome-time-series | 시계열 관련 패키지·논문·학습 자료 폭넓게 모은 Awesome 리스트라 도구 지형 조사에 유용 | [GitHub](https://github.com/lmmentel/awesome-time-series) |
| awesome-time-series (cuge1995) | 연구 논문과 벤치마크, 응용 사례 중심으로 최신 학술 동향 정리한 큐레이션 | [GitHub](https://github.com/cuge1995/awesome-time-series) |
| awesome-industrial-anomaly-detection | 산업 현장 이상탐지 관련 논문·데이터셋·방법론 모은 큐레이션 | [GitHub](https://github.com/M-3LAB/awesome-industrial-anomaly-detection) |
| ts-anomaly-benchmark | 딥러닝 기반 시계열 이상 탐지 방법론·데이터셋·평가 메트릭 모은 벤치마크 (Monash/Griffith/IBM) | [GitHub](https://github.com/zamanzadeh/ts-anomaly-benchmark) |
| awesome-TS-anomaly-detection | 시계열 이상 탐지를 위한 도구·데이터셋 모은 종합 카탈로그 | [GitHub](https://github.com/rob-med/awesome-TS-anomaly-detection) |

## Leaderboards

| Leaderboard | Focus | Link |
| --- | --- | --- |
| Forecasting Experts' Verdict (FEV) | AutoGluon 팀이 운영하는 시계열 예측 리더보드로 다양한 데이터셋 성능 비교 가능 | [Hugging Face Space](https://huggingface.co/spaces/autogluon/fev-leaderboard) |
| GIFT Evaluation Leaderboard | Salesforce 글로벌 시계열 벤치마크 GIFT 평가 결과와 파이프라인 제공 | [Hugging Face Space](https://huggingface.co/spaces/Salesforce/GIFT-Eval) |
