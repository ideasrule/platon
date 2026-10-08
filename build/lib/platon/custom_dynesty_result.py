import dynesty.results
import copy

class CustomDynestyResult:
    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __contains__(self, key):
        return key in self.__dict__

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        return repr(self.__dict__)

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        """
        Return the list of items in the results object as list of key,value pairs
        """
        return ((k, getattr(self, k)) for k in self.__dict__)
    
    def __init__(self, result):
        assert(type(result) == dynesty.results.Results)
        
        for key in result.keys():
            self.__dict__[key] = copy.deepcopy(result[key])
