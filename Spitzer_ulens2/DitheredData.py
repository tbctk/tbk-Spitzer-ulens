import numpy as np

class DitheredData(object):
    def __init__(self,arr):
        self.arr = arr
        self.ndit = len(arr)
        self.shapes = [np.shape(a) for a in arr]
        
    @classmethod
    def from_flat(cls,arr,breaks,labels=None):
        tmp = []
        tmp.append(arr[:breaks[1]])
        for i in range(1,len(breaks)-1):
            tmp.append(arr[breaks[i]:breaks[i+1]])
        return cls(tmp,labels=labels)
    
    def ditherwise(self,func):
        out = []
        for d in self.arr:
            out.append(func(d))
        return out
    
    def __add__(self,d):
        if isinstance(d,DitheredData):
            if d.shapes == self.shapes:
                new_arr = self.arr.copy()
                for i in range(self.ndit):
                    new_arr[i] = np.add(new_arr[i],d.arr[i])
                return DitheredData(new_arr)
            else:
                raise Exception('Error: size mismatch in DitheredData.__add__() method.')
        else:
            raise Exception('Error: type mismatch in DitheredData.__add__() method.')
        return
            
    def sum(self,axis=None):
        if axis is None:
            return sum(arr)
        elif isinstance(axis,int):
            new_arr = self.arr.copy()
            
            
    
    def __str__(self):
        return self.arr.__str__()