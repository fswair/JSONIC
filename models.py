from . import JSONField, Column

@JSONField
class DataModel(object):
    name: str = Column(str)
    ssn: int = Column(int)
    age: int = Column(int)
    price: float = Column(float)
    year: int = Column(int)
@JSONField
class WorkerModel(object):
    ad: str = Column(str)
    soyad: str = Column(str)


# print("%s:" % (DataModel.__name__), DataModel.schema(), sep="\n", end="\n\n")