from typing import Callable, TypeVar, Union

T = TypeVar('T')
def or_none(type_constructor: Callable[[str], T]) -> Callable[[str], Union[T, None]]:
    """
    Wrapper of type_constructor that accept string. If the input
    string makes the type_constructor raise exception, return None.
    """
    def wrapper(s: str) -> Union[T, None]:
        try:
            return type_constructor(s)
        except Exception:
            return None
    return wrapper

T = TypeVar('T')
def or_default(type_constructor: Callable[[str], T], default: T) -> Callable[[str], T]:
    """
    Wrapper of type_constructor that accept string. If the input
    string makes the type_constructor raise exception, return default.
    """
    def wrapper(s: str) -> T:
        try:
            return type_constructor(s)
        except Exception:
            return default
    return wrapper
