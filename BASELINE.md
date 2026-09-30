# CIFAR FedAvg / FedConst 기본 실험

현재 `master`의 FedConst는 별도 모듈이 아니라 FedAvg의 `--const true` 경로입니다. 따라서 이 기본 세트는 같은 FedAvg aggregation 아래에서 `const=false`와 `const=true`를 비교합니다.

환경 생성:

```bash
conda env create -f environment.yml
conda activate fedtorch
```

전체 2 × 2 × 2 기본 매트릭스 실행:

```bash
bash scripts/run_baselines.sh --gpu true --rounds 50
```

CPU smoke run은 `--gpu false --rounds 1 --clients 2 --local-epochs 1`을 사용합니다. 본 비교는 `cifar-10/cifar-100`, `Custom_cnn/resnet-18`, FedAvg/FedConst, 10 clients, Dirichlet α=0.5, SGD (lr 0.01, momentum 0.9, wd 1e-4), local epoch 5, 50 global rounds를 기본으로 사용합니다.

반복 실험 예시:

```bash
bash scripts/run_baselines.sh --gpu true --rounds 200 --seeds 2023,2024,2025
```

각 실행은 `logs/` 아래에 저장되며, 배치 종료 후 `logs/<tag>_summary.csv`에 마지막 공식 CIFAR test accuracy를 모읍니다. 기본 실행에서는 오래 걸리고 CPU에서 실패할 수 있는 Hessian/loss-landscape diagnostics를 끕니다.

기본 스크립트는 `--save_model false --save_data false`로 실행하므로 모델 체크포인트와 데이터 스냅샷을 저장하지 않습니다.
