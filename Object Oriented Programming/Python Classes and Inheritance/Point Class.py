import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # def getx(self):
    #     return self.x

    # def gety(self):
    #     return self.y

    def distanceFromOrigin(self):
        return math.sqrt(self.x ** 2 + self.y**2)

    def distancebetweenpoints(self,x,y):
        dist_x = x - self.x
        dist_y = y - self.y
        dist = math.sqrt(dist_x**2 + dist_y**2)
        return dist

    def __str__(self):
        return 'x={} y={}'.format(self.x, self.y)



p1 = Point(7,6)
print(p1)
print(p1.x, p1.y)
print(p1.distanceFromOrigin())
print(p1.distancebetweenpoints(6,3))