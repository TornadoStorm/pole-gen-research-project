from typing import Tuple

import torch.utils.data as data

from models.data_source_info import DataSourceInfo


class DataSource:
    def download(self) -> Tuple[DataSourceInfo, data.Dataset, data.Dataset]:
        pass
