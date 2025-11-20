#%%
from collections import abc
import numbers
#%%
# Declare point descriptor
class ValidatePoints:
    def __init__(self,min_value=None, max_vaue=None):
        self.min_value = min_value
        self.max_vaue = max_vaue

    def __set_name__(self,owner_class,name):
        self.name = name

    def __set__(self,instance,value):
        if not isinstance(value,numbers.Real):
            raise TypeError(f'{self.__class__.__name__}:Values for {self.name} must be Real.')
        if abs(value) < self.min_value:
            raise ValueError(f'{self.__class__.__name__}:Value for {self.name} cannot be less than {self.min_value} value')
        if abs(value) > self.max_vaue:
            raise ValueError(f'{self.__class__.__name__}:Value for {self.name} cannot be more than {self.max_vaue}')
        instance.__dict__[self.name] = value

    def __get__(self,instance,owner_class):
        return instance.__dict__[self.name]
#%%
class Point2D:
    x = ValidatePoints(min_value = 0, max_vaue=800)
    y = ValidatePoints(min_value = 0, max_vaue=800)

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f'({self.x}, {self.y})'
#%%
#Define a descriptor to validate the polygon
class Point2DSequence:
    def __init__(self,length=None):
        self.length = length
       
    def __set_name__(self,cls,name):
        self.name = name

    def __set__(self, instance, value):

               
        if not isinstance(value,abc.Sequence):
            raise TypeError(f'{self.__class__.__name__}:{self.name} must be of type sequence.')
            
        if self.length is None :
            raise ValueError(f'{self.__class__.__name__}:{self.name} must specify the number of vertices.')

        if len(value) != self.length:
            raise ValueError(f'{self.__class__.__name__}:{len(value)} points have been specified. {self.length} are needed')

        list_of_vertices = []
        
        for index, item in enumerate(value):
           
            if isinstance(item, Point2D):
                list_of_vertices.append(item)
                
            elif (
                (isinstance(item, abc.Sequence)) 
                and len(item) == 2 
                ):
                  list_of_vertices.append(Point2D(*item))
            else:
                 raise TypeError(f'{self.__class__.__name__}:Element at index {index} must be Point2D or a (x,y) pair')
      
                          
        instance.__dict__[self.name] = list_of_vertices

    def __get__(self, instance, cls):
        return instance.__dict__[self.name]
    
#%%
class Polygon:
    vertices = Point2DSequence(4)

    def __init__(self, *vertices):
        self.vertices = vertices

    def __repr__(self):
        return f'{self.vertices}'
#%%
class Triangle(Polygon):
    vertices=Point2DSequence(3)
    
#%%
try:
    t = Triangle((20,300),(30,40),(0,50))
except TypeError as ex1:
    print(ex1)
except ValueError as ex2:
    print(ex2)
 

#%%
class Quadrilateral(Polygon):
    vertices = Point2DSequence(4)
#%%
q = Quadrilateral((0,30),(0,40),(0,100),(3,40))
#%%
t, q
#%%
