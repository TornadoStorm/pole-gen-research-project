import json
from typing import List


class DataSourceInfo:
    class_names: List[str]

    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        pass

    def to_dict(self):
        return {"class_names": self.class_names}


class DataSourceInfoEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, DataSourceInfo):
            return obj.to_dict()
        return super().default(obj)
