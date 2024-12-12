class DataSource:
    url: str
    name: str


class ModelNet40(DataSource):
    url = "http://modelnet.cs.princeton.edu/ModelNet40.zip"
    name = "ModelNet40"
    checksum = "42dc3e656932e387f554e25a4eb2cc0e1a1bd3ab54606e2a9eae444c60e536ac"
