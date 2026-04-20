from functools import wraps
from pydantic import BaseModel


def gather_params(querycls: type[BaseModel]):
    def _accept_query(func):
        @wraps(func)
        def inner(self, **kwargs):
            if len(kwargs) == 1 and 'params' in kwargs:
                return func(self, params=kwargs['params'])
            else:
                query_obj = querycls.model_validate(kwargs)
                return func(self, params=query_obj)
        return inner
    return _accept_query
