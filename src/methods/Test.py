import os
import torch
import ray
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from src.model import NUMBER_OF_CLASSES
from .utils import *
from . import *


# 다른 필요한 종속성들도 추가될 수 있습니다.

@ray.remote(max_calls=1)
def compute_hessian_for_model(client: Client, summary_path: str):
    """
    주어진 모델에 대해 헤시안 값을 계산하고 텐서보드에 기록합니다.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = client.model.to(device)

    # 텐서보드에 기록하기 위한 SummaryWriter를 초기화합니다.
    summary_writer = SummaryWriter(summary_path)

    # F.mark_hessian 함수는 모델, 데이터로더, 텐서보드의 writer, 그리고 global round를 입력받아 헤시안 값을 계산하고 기록합니다.
    # 여기서는 테스트 데이터로더를 사용합니다.
    F.mark_hessian(model, client.test_loader, summary_writer, client.epoch_counter)

    summary_writer.close()

    return True


def test_hessian_on_saved_model(client_setting: dict, model_path: str):
    stream_logger, _ = get_logger(LOGGER_DICT['stream'])
    summary_logger, _ = get_logger(LOGGER_DICT['summary'])

    # 저장된 모델을 불러옵니다.
    saved_model = torch.load(model_path)

    # INFO - Dataset creation
    _, _, test_loader = data_preprocessing(client_setting)

    client = Client
    _, aggregator = client_initialize(client, None, None, test_loader, None,
                                      client_setting, None)

    aggregator.model.load_state_dict(saved_model)

    # 헤시안 값을 계산하고 텐서보드에 기록합니다.
    compute_hessian_for_model.remote(aggregator, os.path.join(aggregator.summary_path, "hessian"))

    stream_logger.info("Hessian values have been computed and recorded in Tensorboard.")

# 실제로 위의 함수를 호출하려면 다음과 같이 할 수 있습니다.
# client_setting 및 model_path는 실제 설정에 따라 변경되어야 합니다.
# client_setting = {...}
# model_path = "path_to_saved_model.pth"
# test_hessian_on_saved_model(client_setting, model_path)
