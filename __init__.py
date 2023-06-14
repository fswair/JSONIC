from typing import TypeVar, Union, Callable, Any, Type, TextIO, Generator
import os, re, json, sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

types = dict(int="INTEGER", str="STRING", float="FLOAT")

Self = TypeVar("Self", object, bytes)

class StaticResponse(object):
    __doc__ = "Response Filler Class for @jsonic"

class Where(object):
    def __init__(cls, conn:  "JSONIC", _filter: Callable = lambda x: bool(x) == True):
        cls.filter = _filter
        cls.conn = conn
        cls.datas = cls.conn.get()
        cls.response_array = np.fromiter(filter(cls.filter, cls.datas), np.dtype)
    def __str__(cls) -> str:
        return str(cls.response_array.tolist())
    def __repr__(cls) -> str:
        return cls.__str__()
    def first(cls) -> dict[Union[int, str], Union[int, str, list, dict]]:
        return cls.response_array[0]
    def all(cls):
        return cls.response_array
    def sliced(cls, __slice: slice):
        return cls.response_array[__slice]
    def to_model(cls) -> list[object]:
        models = list()
        for response in cls.response_array.tolist():
            response: dict
            model = StaticResponse()
            for k,v in response.items():
                setattr(model, k, v)
            models.append(model)
        return models
    def to_dataframe(cls, print_to_stdout: bool = False, _index: bool = ...) -> pd.DataFrame:
        data = cls.response_array.tolist()
        frame = pd.DataFrame(data)
        if print_to_stdout:
            print(frame)
        return frame

    def to_json(cls):
        return cls.to_dataframe().to_json()
    
    def to_csv(cls):
        return cls.to_dataframe().to_csv()
    
    def to_excel(cls, _excel_fpath: Union[str, Path]):
        assert (_excel_fpath), "This path value does not belong a valid path."
        return cls.to_dataframe().to_excel(excel_writer=_excel_fpath)
    
    def to_sql(cls, __name: str, __conn: sqlite3.Connection):
        return cls.to_dataframe().to_sql(__name, __conn)
    
    def to_markdown(cls):
        return cls.to_dataframe().to_markdown()
    
    def to_html(cls):
        return cls.to_dataframe().to_html()
    
    def to_json(cls, path: Union[str, Path] = ..., buffer: TextIO = ...):
        return cls.to_dataframe().to_json(path or buffer)
    
    def to_xml(cls, path: Union[str, Path] = ..., buffer: TextIO = ...):
        return cls.to_dataframe().to_xml(path or buffer)
    
    def to_numpy(cls, filter: Callable = None) -> np.ndarray:
        array = cls.to_dataframe().to_numpy(np.dtype)
        if filter:
            arr = (i for i in array if filter(i))
        else:
            arr = (i for i in array)
        return np.fromiter(arr)
            


class JSONField():
    def __schema__(cls, instance: Callable, datafields: list[str]):
        data = list()
        for field in datafields:
            typer: object = instance.__dict__[field]
            data.append(dict(field=field, typer=typer))
        return data
            
    def __new__ (cls, instance: Callable) -> object:
        cls.datafields = list(filter(lambda q: not q.startswith("__"), dir(instance)))
        cls.datafields.sort()

        setattr(instance, "schema", lambda: cls.__schema__(cls, instance, cls.datafields))
        setattr(instance, "where", lambda conn, filter: cls.where(cls, conn, filter))

        return instance
    
    def where(cls, conn: "JSONIC" = ..., filter: Callable = ...) -> Where:
        return Where(conn, _filter=filter)

    def is_formattable_to(cls, __obj: object, __type: Type):
        res = None
        try:
            res = __type(__obj)
        except TypeError as e:
            raise Exception(f"{__obj.__class__.__name__} instance can not formatted as {__type.__name__}.")
        return res

class JSONIC:
    def loads(self: Self): return json.loads
    def load(self: Self):  return json.load
    def dumps(self: Self): return json.dumps
    def dump (self: Self): return json.dump
    def __init__(cls, __json: Union[str, Path] = "./jsonic/database.json", model: JSONField = None, mode: str = "r+"):
        cls.fpath = Path(__json)
        cls.mode = mode
        cls.model = model
        module = __import__("json")
        for attr in dir(module):
            setattr(cls, attr, getattr(module, attr))
    
    def table(cls):
        return Table(cls)
    
    def close(cls):
        return cls.fbinary.close()
    
    def write(cls, __data: dict) -> dict:
        data: dict[str, list] = cls.read()
        data["data"].append(__data)
        file = open(cls.fpath, "w+", encoding="utf8")
        file.write(cls.dumps(data))
    
    def get(cls) -> list[dict]:
        data = cls.read()
        return data["data"]

    
    def read(cls, encoding: str = "utf8")  -> dict:
        if cls.data:
            data = cls.loads(cls.data)
            return data
        return {}

    def __enter__(cls):
        assert (cls.model is not None), Exception("You have to pass JSONField model.")
        if not os.path.isfile(cls.fpath):
            cls._temp_binary = open(cls.fpath, "w+")
            table = cls.model.__name__.rstrip("Model").title()
            cls._temp_binary.write(cls.dumps(dict(table=table, data=[])))
            cls._temp_binary.close()
        cls.fbinary = open(cls.fpath, cls.mode)
        cls.data = cls.fbinary.read()
        return cls
    def __exit__(cls, exc_type: Union[Any, None], exc_val: Union[Any, None], traceback: Union[Any, None]):
        return cls.fbinary.close()
    
    async def __aenter__(cls):
        if not os.path.isfile(cls.fpath):
            cls._temp_binary = open(cls.fpath, "w+")
            cls._temp_binary.write("{}")
            cls._temp_binary.close()
        cls.fbinary = open(cls.fpath, cls.mode)
        return cls
    async def __aexit__(cls, exc_type: Union[Any, None], exc_val: Union[Any, None], traceback: Union[Any, None]):
        return cls.fbinary


class Typer:
    def __new__(cls, typer: dict[str, str]):
        cls.finder = lambda __stack: re.search("(INTEGER|STRING|FLOAT)", __stack, re.I).group()
        cls.name = list(typer.keys())[0]
        cls.typer = typer.get(cls.name)

        match (cls.finder(cls.typer)):
            case "STRING":
                cls.typer = dict(name=cls.name, typer=str)
            case "INTEGER":
                cls.typer = dict(name=cls.name, typer=int)
            case "FLOAT":
                cls.typer = dict(name=cls.name, typer=float)
        return cls.typer

class FilterItem(object):
    def __new__(cls, filter: Callable, key: bool = False, value: bool = True) -> Self:
        cls.filter = filter
        cls.key = key
        cls.value = value

        return cls

class Stack(object):
    def __init__(cls, table: "Table", conn: "JSONIC", data: dict) -> None:
        cls.table = table
        cls.conn = conn
        cls.data = data
    
    def commit(cls):
        datafields = list(filter(lambda q: not q.startswith("__"), dir(cls.conn.model)))
        datafields.pop(datafields.index("schema"))
        datafields.pop(datafields.index("where"))
        print(datafields, cls.data.items().__len__() == datafields.__len__())
        if cls.data.items().__len__() == datafields.__len__():
            data_keys = list(cls.data.keys()).sort()
            model_keys = datafields.sort()
            print(model_keys == data_keys)
            if model_keys == data_keys:
                cls.conn.write(cls.data)
                cls.data = dict()
                table = cls.conn.table()
                cls.stack = Stack(table, cls.conn, cls.data)
                return cls.conn.model.where(cls.conn, filter=bool).to_model()
        else:
            return []

    def __setitem__(cls, __item: str | int, __value: Any):
        cls.data[__item] = __value
        return cls.data[__item]

    def __getitem__(cls, item: str | FilterItem) -> Generator | dict:
        res = dict()
        responses = list()
        if isinstance(item, FilterItem):
            for data in cls.conn.get():
                res = dict()
                for k,v in data.items():
                    if item.filter(v if item.value else k):
                        res[k] = v
                responses.append(res)
            return responses
        if type(item) in (int, str):
            print(item)
            return cls.data[item]
        return {}

class Table(object):
    __name__: str = "@jsonic"
    def __new__(cls, conn: "JSONIC") -> Self:
        cls.data = dict()
        cls.conn = conn
        cls.stack = Stack(cls, conn, cls.data)
        return cls
        

class Column(object):
    def __new__(cls, typer: Typer = None):
        assert (typer and typer is not None), "Please pass typer instance"
        cls.typer = typer
        return cls.typer

class ConfigSerializer(object):
    name: Table
    fields: list[Typer]
    def __new__(cls):
        with JSONIC() as cls.jsonic:
            for k, v in cls.jsonic.data.items():
                setattr(cls, k, v)
            cls.typers = list(map(Typer, cls.fields))
            return cls