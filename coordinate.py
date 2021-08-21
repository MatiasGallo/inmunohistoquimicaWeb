class Coordinate(object):
    '''Creates a point on a coordinate plane with values x and y.'''

    def __init__(self, x, y):
        '''Defines x and y variables'''
        self.X = x
        self.Y = y

    def getX(self):
        return self.X

    def getY(self):
        return self.Y

    def __repr__(self):
        return str(self.X) + ',' + str(self.Y)
    def __str__(self):
        return str(self.X) + ',,,' + str(self.Y)