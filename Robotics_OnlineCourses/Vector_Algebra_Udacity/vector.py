class Vector(object):
    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple(coordinates)
            self.dimension = len(coordinates)

        except ValueError:
            raise ValueError('The coordinates must be nonempty')

        except TypeError:
            raise TypeError('The coordinates must be an iterable')


    def __str__(self):
        return 'Vector: {}'.format(self.coordinates)


    def __eq__(self, v):
        return self.coordinates == v.coordinates

    def plus(v1, v2):
        try:
            if not len(v1) == len(v2):
                raise ValueError
            return [v1[i] + v2[i] for i in range(len(v1))]

                
        except ValueError:
            raise ValueError('Vectors must have the same lenght')
