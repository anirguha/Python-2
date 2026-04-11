#%%
#Define two decriptors - 
#1. For Integer Validation
#2. For Character Validation
#%%
from numbers import Integral
#%%
#Declare descriptor to validate both integers and charcarters
class BaseField:
    def __init__(self, min_=None, max_=None):
        self.min = min_
        self.max = max_

    def __set_name__(self, owner_class, name):
        self.name = name

    def validate_value(self, value):
        pass
        
    def __set__(self,instance,value):
        self.validate_value(value)
        instance.__dict__[self.name]=value

    def __get__(self,instance,cls):
        if instance is None:
            return self
        return instance.__dict__[self.name]
#%%
#Declare descriptor to validate integer inheritating from BaseField
class IntegerField(BaseField):
    def validate_value(self,value):
        if not isinstance(value, Integral):
            raise TypeError(f'{self.name} must be of type int.')
        if self.min is not None and value < self.min:
            raise ValueError(f'{self.name} must be greater than {self.min}')
        if self.max is not None and value > self.max:
            raise ValueError(f'{self.name} must be less than {self.max}')
      
#%%
#Declare descriptor to validate string
class CharField(BaseField):
    def __init__(self,min_,max_):
        if min_ is None or min_<=0:
            min_=0
        super().__init__(min_, max_)

    def validate_value(self, value):   
        if not isinstance(value, str):
            raise TypeError(f'{self.name} must be of type string.')
        if self.min is not None and len(value) < self.min:
            raise ValueError(f'{self.name} must be of minimum length {self.min}')
        if self.max is not None and len(value) > self.max:
            raise ValueError(f'{self.name} must be of maximum length {self.max}')
        
#%%
class Person:
    name = CharField(3, 30)
    age = IntegerField(10,50)

#%%
p = Person()
p.age = 44
p.name = 'Guido'
#%%
import unittest
#%%
#Setting up of unit test suite
def run_tests(test_cls):
    suite = unittest.TestLoader().loadTestsFromTestCase(test_cls)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
#%%
#Test for IngerField descriptor
class TestIntegerField(unittest.TestCase):

    #Define a static method and create dynamic class with attributes
    #based on the decriptor class

    @staticmethod
    def create_test_class(min_,max_):
        obj = type('TestClass',(),{'age':IntegerField(min_,max_)})
        return obj()

    def test_set_age_ok(self):
        """Test for valid values"""
        min_=5
        max_=10
        obj = self.create_test_class(min_,max_)

        valid_values = range(min_,max_+1)
        for idx, value in enumerate(valid_values):
            with self.subTest(test_number=idx):
                obj.age = value
                self.assertEqual(value, obj.age)

    def test_set_age_invalid(self):  
        """Test for invalid values raises appropriate error values"""
        min_= -20
        max_= 20
        obj = self.create_test_class(min_,max_)
        
        bad_values = list(range(min_-5,min_)) #test for min values
        bad_values += list(range(max_+1, max_+5)) #test for max values
     
          
        for idx, value in enumerate(bad_values):
            with self.subTest(test_number=idx):         
                with self.assertRaises(ValueError):
                    obj.age = value

    def test_set_age_invalid_types(self):  
        """Test for invalid values raises appropriate error types"""
        min_= 5
        max_= 20
        obj = self.create_test_class(min_,max_)
        
        invalid_type_values = ['abc', 10.5, (1,2)] #test for invalid types
        for idx, value in enumerate(invalid_type_values):
            with self.subTest(test_number=idx):         
                with self.assertRaises(TypeError):
                    obj.age = value

    def test_set_min_only(self):
        """Test when only the min value is set"""
        min_=10
        max_=None
        obj = self.create_test_class(min_,max_)

        values = range(min_,min_+100,10)
        for idx, value in enumerate(values):
            with self.subTest(test_number=idx):
                obj.age = value
                self.assertEqual(value, obj.age)

    def test_set_max_only(self):
        """Test when only the max value is set"""
        min_=None
        max_=100
        obj = self.create_test_class(min_,max_)

        values = range(max_-100,max_,10)
        for idx, value in enumerate(values):
            with self.subTest(test_number=idx):
                obj.age = value
                self.assertEqual(value, obj.age)

    def test_set_nolimit(self):
        """Test when only there is no limit to age"""
        min_=None
        max_=None
        obj = self.create_test_class(min_,max_)

        values = range(-100,100,10)
        for idx, value in enumerate(values):
            with self.subTest(test_number=idx):
                obj.age = value
                self.assertEqual(value, obj.age)         
#%%
run_tests(TestIntegerField)
#%%
#Test for CharField descriptor
class TestCharField(unittest.TestCase):

    #Define a static method and create dynamic class with attributes
    #based on the decriptor class

    @staticmethod
    def create_test_class(min_,max_):
        obj = type('TestClass',(),{'name':CharField(min_,max_)})
        return obj()

    def test_set_name_ok(self):
        """Test for valid string length values can be used"""
        min_=1
        max_=10
        obj = self.create_test_class(min_,max_)

        valid_values = range(min_,max_+1)
        for idx, value in enumerate(valid_values):
            name_value = 'a'* value
            with self.subTest(test_number=idx):
                obj.name = name_value
                self.assertEqual(name_value, obj.name)

    def test_set_name_invalid_types(self):  
        """Test for invalid types raises appropriate error values"""
        min_= -20
        max_= 20
        obj = self.create_test_class(min_,max_)
        
        bad_values = [1, 1.6, (2,3)] #test for ivalid types   
          
        for idx, value in enumerate(bad_values):
            with self.subTest(test_number=idx):         
                with self.assertRaises(TypeError):
                    obj.name = value

    def test_set_name_invalid_name_length(self):  
        """Test for invalid name lengths raises appropriate error values"""
        min_= 5
        max_= 20
        obj = self.create_test_class(min_,max_)
        
        bad_lengths = [0, 1, 4, 21, 30, 35] 
          
        for idx, value in enumerate(bad_lengths):
            name_value = 'a' * value
            with self.subTest(test_number=idx):         
                with self.assertRaises(ValueError):
                    obj.name = name_value

    def test_set_min_only(self):
        """Test when only the min value is set"""
        min_=10
        max_=None
        obj = self.create_test_class(min_,max_)

        values = range(min_,min_+100,10)
        for idx, value in enumerate(values):
            name_value = 'a' * value
            with self.subTest(test_number=idx):
                obj.name = name_value
                self.assertEqual(name_value, obj.name)

    def test_set_max_only(self):
        """Test when only the max value is set"""
        min_=None
        max_=100
        obj = self.create_test_class(min_,max_)

        values = range(max_-100,max_,10)
        for idx, value in enumerate(values):
            name_value = 'a' * value
            with self.subTest(test_number=idx):
                obj.name = name_value
                self.assertEqual(name_value, obj.name)

    def test_set_nolimit(self):
        """Test when only there is no limit to name length"""
        min_=None
        max_=None
        obj = self.create_test_class(min_,max_)

        values = [v for v in range(-100,100,10) if v>=0]
        for idx, value in enumerate(values):
            name_value = 'a' * value
            with self.subTest(test_number=idx):
                obj.name = name_value
                self.assertEqual(name_value, obj.name)   
#%%
run_tests(TestCharField)