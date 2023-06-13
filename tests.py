from . import JSONIC
from models import DataModel, JSONField

with JSONIC() as jsonic:
    DataModel: JSONField
    state = DataModel.where(conn=jsonic, filter=lambda data: data["id"] == 1001)
    print("Response:\n", state)