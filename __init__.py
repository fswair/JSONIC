from typing import TypeVar, Union, Callable, Any
import os, re, json as JSON, numpy as np
from pathlib import Path

types = dict(int="INTEGER", str="STRING", float="FLOAT")

Self = TypeVar("Self", object, bytes)



class Where(object):
    def __new__(cls, conn:  "JSONIC", filter: Callable = lambda x: bool(x) == True):
        cls.filter = filter
        cls.conn = conn
        cls.datas = cls.conn.get()
        cls.response_array = np.fromiter((data for data in cls.datas if filter(data)), np.dtype)
        return cls.response_array.tolist()

class JSONField(object):
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
        

    def __call__(cls):
        return cls
    
    def where(cls, conn: "JSONIC" = ..., filter: Callable = ...) -> list[dict]:
        return Where(conn, filter=filter)

class JSONIC:
    def loads(self: Self):  return JSON.loads
    def load(self: Self):  return JSON.load
    def dumps(self: Self):  return JSON.dumps
    def dump (self: Self):  return JSON.dump
    def __init__(cls, __json: Union[str, Path] = "./JSONIC/database.json", mode: str = "r+"):
        cls.fpath = Path(__json)
        cls.mode = mode
        module = __import__("json")
        for attr in dir(module):
            setattr(cls, attr, getattr(module, attr))
    
    def write(cls) -> dict:
        return cls
    
    def get(cls) -> list[dict] | list:
        return [dict(id=1001, order=999), dict(id=1001, order=998), dict(id=1000, order=997)]
    
    def read(cls, encoding: str = "utf8")  -> dict:
        if cls.fbinary.read():
            return cls.load(cls.fbinary)
        return {}
    
    @property
    def data(cls):
        return cls.read()

    def __enter__(cls):
        if not os.path.isfile(cls.fpath):
            cls._temp_binary = open(cls.fpath, "w+")
            cls._temp_binary.write("{}")
            cls._temp_binary.close()
        cls.fbinary = open(cls.fpath, cls.mode)
        return cls
    def __exit__(cls, exc_type: Union[Any, None], exc_val: Union[Any, None], traceback: Union[Any, None]):
        return cls.fbinary.close()


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

class Table(object):
    __name__: str = "@jsonic"

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