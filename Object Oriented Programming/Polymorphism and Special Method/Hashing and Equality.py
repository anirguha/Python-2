#%%
#Hash and Equality
class Person:
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, Person) and self.name == other.name

    def __repr__(self):
        return f'{self.name}'
#%%
p1 = Person('Eric')
p2 = Person('Eric')
p3 = Person('Amanda')

p1==p2, hash(p1) == hash(p2)
# p1==p3, hash(p1) == hash(p3)?
#%%
d = {p1: 'Eric'}
#%%
d
#%%
