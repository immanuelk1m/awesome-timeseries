# Awesome Time Series Forecast

> 시계열 예측·분석 위한 **프레임워크, 에이전트, 라이브러리, 큐레이션 모음**. (실무 · 대회 · 연구 어디서든 바로 써먹을 링크 제공 목표.)

`awesome-time-series` 레포 정리 스타일 참고해 큐레이션 재구성. 요청 따라 **모델 / 에이전트 / 학습자료 / 리더보드** 체계 사용.

## Contents
- [Models](#models) 목록
- [Agents](#agents) 항목
- [Papers](#papers) 논문
- [Learning Resources](#learning-resources) 참고용
- [Utilities](#utilities) 도구
- [Leaderboards](#leaderboards) 정리
- [Community](#community) 토론
- [Related References](#related-references) 큐레이션

## Models

| Model | Description | Paper | GitHub | 비고 |
| --- | --- | --- | --- | --- |
| TimesFM 2.5 200M | 다중 도메인 시계열용 파운데이션 Transformer 모델 | [arXiv:2310.10688](https://arxiv.org/abs/2310.10688) | [google-research/timesfm](https://github.com/google-research/timesfm) | Foundation Transformer |
| Autoformer | Auto-Correlation 기반 자동 분해로 장기 예측 성능 높인 모델 | [arXiv:2106.13008](https://arxiv.org/abs/2106.13008) | [thuml/Autoformer](https://github.com/thuml/Autoformer) | Transformer |
| FEDformer | Fourier/Wavelet 기반 주파수 분해로 장기 예측 효율 높인 모델 | [ICML 2022](https://proceedings.mlr.press/v162/zhou22g/zhou22g.pdf) | [MAZiqing/FEDformer](https://github.com/MAZiqing/FEDformer) | Transformer |
| Informer | ProbSparse Attention으로 긴 시계열 효율적으로 모델링하는 구조 | [arXiv:2012.07436](https://arxiv.org/abs/2012.07436) | [zhouhaoyi/Informer2020](https://github.com/zhouhaoyi/Informer2020) | Transformer |
| iTransformer | 변수를 토큰으로 다루는 Inverted Transformer로 다변량 상관관계 학습 | [OpenReview](https://openreview.net/pdf?id=QK6ogQ7MZLX) | [thuml/iTransformer](https://github.com/thuml/iTransformer) | Transformer |
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
| DLinear | 분해 + 선형 회귀 활용해 간결하면서 강력한 예측 베이스라인 제공 | [arXiv:2205.13504](https://arxiv.org/abs/2205.13504) | [vivva/DLinear](https://github.com/vivva/DLinear) | Linear |
| DeepEDM | 시계열 동역학 명시적으로 학습하는 DeepEDM 접근법 | [Project Page](https://abrarmajeedi.github.io/deep_edm/) | - | Dynamics |

## Agents

| Agent | Description | GitHub |
| --- | --- | --- |
| DeepAnalyze | 데이터 중심 업무(EDA→전처리→모델링→리포트)를 에이전트형 LLM이 자율 처리하도록 설계 | [ruc-datalab/DeepAnalyze](https://github.com/ruc-datalab/DeepAnalyze) |
| TimeCopilot | LLM과 Chronos/Moirai/TimesFM/TimeGPT 백엔드 결합해 예측·교차검증·이상탐지 자동화 | [AzulGarza/timecopilot](https://github.com/AzulGarza/timecopilot) |

## Papers

| Title | Description | Link |
| --- | --- | --- |
| arXiv:2510.02729 | 최신 시계열 관련 프리프린트 (세부 내용은 원문 참고) | [arXiv:2510.02729](https://arxiv.org/pdf/2510.02729) |

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

## Leaderboards

| Leaderboard | Focus | Link |
| --- | --- | --- |
| Forecasting Experts' Verdict (FEV) | AutoGluon 팀이 운영하는 시계열 예측 리더보드로 다양한 데이터셋 성능 비교 가능 | [Hugging Face Space](https://huggingface.co/spaces/autogluon/fev-leaderboard) |
| GIFT Evaluation Leaderboard | Salesforce 글로벌 시계열 벤치마크 GIFT 평가 결과와 파이프라인 제공 | [Hugging Face Space](https://huggingface.co/spaces/Salesforce/GIFT-Eval) |

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
