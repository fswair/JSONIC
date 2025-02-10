"""
Library types for type hinting and dependency specification
"""

import datetime
import enum
import random
from functools import partial
from types import NoneType
from typing import TypeAlias, TypeVar, Generic, Optional, Any, Callable, final
from abc import ABC, abstractmethod

Self = TypeVar("Self")
Typer = TypeVar("Typer", bound=type)
T = TypeVar("T")
NULL: TypeAlias = None
Missing = TypeVar("Missing", bound=NoneType)

@final
class Random(ABC):
    """A utility class providing static methods for generating random values.
    This class contains methods for generating random numbers within a range,
    random dates, and random hash values.
    """
    
    @abstractmethod
    def __init__(self): pass
    
    @staticmethod
    def setrange(start: int, stop: int, step: int = 1) -> Callable[[], int]:
        """
        Creates a callable function that returns a random integer from a specified range.

        ## Args:
            start (int): The starting value of the range (inclusive).
            stop (int): The ending value of the range (exclusive).
            step (int, optional): The step/increment between numbers in the range. Defaults to 1.

        ## Returns:
            Callable[[], int]: A function that when called returns a random integer from the specified range.

        ## Example:
            >>> random_num = setrange(0, 10, 2)  # Creates range [0,2,4,6,8]
            >>> random_num()  # Returns random value from the range
            6
        """
        return partial(
            random.choice,
            range(start, stop, step)
        )
        
    
    @staticmethod
    def date(date_fmt: Optional[str] = None) -> str:
        """
        Generate a random datetime between 1970-01-01 and current time.

        ## Args:
            date_fmt (Optional[str]): Format string for the returned date. Defaults to '%d-%m-%Y %H:%M' if None.

        ## Returns:
            str: A formatted string representation of the random datetime.

        ## Examples:
            >>> Random.date()  # returns e.g. '15-06-1995 13:24'
            >>> Random.date('%Y-%m-%d')  # returns e.g. '1995-06-15'
        """
        now = datetime.datetime.now()
        start = int(datetime.datetime(1970, 1, 1).timestamp())
        end = int(now.timestamp())
        return datetime.datetime.fromtimestamp(random.randrange(start, end)).strftime(date_fmt or "%d-%m-%Y %H:%M")
    
    @staticmethod
    def hash() -> str:
        """
        Generates a random 128-bit hexadecimal hash string.

        The function uses Python's random module to generate a 128-bit random number,
        converts it to a 16-byte sequence, and returns its hexadecimal representation.

        ## Returns:
            str: A 32-character hexadecimal string representing a random 128-bit hash.

        ## Example:
            >>> hash()
            'a1b2c3d4e5f67890123456789abcdef0'
        
        ## Note:
            - Please do not inherit or instantiate from this class, as it may lead to exceptions.
        """
        return random.getrandbits(128).to_bytes(16, "big").hex()

@final
class Date(ABC):
    """A utility class for date-related operations.

    This class provides static methods for common date operations and formatting.
    """
    
    @abstractmethod
    def __init__(self): pass
    
    timestamp = datetime.datetime.now().timestamp
    
    @staticmethod
    def now(date_fmt: Optional[str] = None) -> str:
        """
        Returns the current date and time formatted as a string.

        ## Args:
            date_fmt (str, optional): The format string for the date/time output. 
                Defaults to "%d-%m-%Y" if not specified.
                Uses datetime.strftime() format codes.

        ## Returns:
            str: The current date/time formatted according to date_fmt.

        ## Examples:
            >>> now()  
            '25-12-2023'
            >>> now('%Y-%m-%d %H:%M:%S')
            '2023-12-25 13:45:30'
        ## Note:
            - Please do not inherit or instantiate from this class, as it may lead to exceptions.
        """
        return datetime.datetime.now().strftime(date_fmt or "%d-%m-%Y")

@final
class Instance(object):
    """
    A response filler class for @jsonic decorator.
    This class serves as a base for converting JSON-like data structures into Python objects.
    It provides string representation methods through __repr__ and __str__.
    The class uses __slots__ = () to prevent dynamic attribute creation and optimize memory usage.
    ## Returns:
        Instance: An instance object that can be used to store JSON data as attributes.
    ## Example:
        >>> inst = Instance()
        >>> inst.name = "test"
        >>> print(inst)
        {'name': 'test'}
    ## Note:
        - Please do not inherit from this class, as it may lead to exceptions.
    """
    __doc__ = "Response Filler Class for @jsonic"
    __slots__ = ()
    
    def __repr__(self) -> str:
        return repr(self.__dict__)
    
    def __str__(self) -> str:
        return str(self.__dict__)

@final
class Object(Generic[T], dict):
    """A dictionary-like object that allows attribute-style access to its items.
    This class inherits from both Generic[T] and dict, providing a way to access dictionary
    items using either dictionary-style (obj['key']) or attribute-style (obj.key) notation.
    Type Parameters:
        T: The type of values stored in the object.
    ## Examples:
        >>> obj = Object()
        >>> obj.name = "John"
        >>> obj['age'] = 30
        >>> print(obj.name)  # Outputs: "John"
        >>> print(obj['age'])  # Outputs: 30
        >>> print(obj)  # Outputs: Object({'name': 'John', 'age': 30})
    ## Note:
        - When accessing attributes, it first checks if the key exists in the dictionary,
        then falls back to normal attribute lookup if it doesn't.
        - Please do not inherit from this class, as it may lead to exceptions.
    """
    def __str__(self) -> str:
        """Returns a string representation of the object."""
        return f"Object({dict(self)})"
    
    def __getattribute__(self, name: str) -> Any:
        """
        Customizes attribute access for dictionary-like objects.

        ## Args:
            name (str): The name of the attribute being accessed.

        ## Returns:
            Any: The value associated with the attribute name if it exists as a key,
             otherwise the result of default attribute lookup.

        ## Example:
            >>> obj = DictLikeClass()
            >>> obj['foo'] = 'bar'
            >>> obj.foo  # Returns 'bar' through __getattribute__
        
        ## Note:
            - This method allows accessing dictionary keys as attributes.
            - Please do not inherit from this class, as it may lead to exceptions.
        """

        if name in self:
            return self[name]
        return super().__getattribute__(name)
    
    def __setattr__(self, name: str, value: Any) -> None:
        """
        Override the attribute setting behavior to store attributes in the dictionary.

        This method allows setting attributes using dot notation, which are then stored
        in the underlying dictionary.

        ## Args:
            name (str): The name of the attribute to set
            value (Any): The value to assign to the attribute

        ## Returns:
            None

        ## Example:
            >>> obj = ClassName()
            >>> obj.new_attr = 123  # Sets obj['new_attr'] = 123
        """
        self[name] = value
    
    def __getitem__(self, key):
        return self.get(key)