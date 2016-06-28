import math
import matplotlib.pyplot as plt

def rtpairs(radii, numbers):
    """
    Given a list of radii and a list of quantities,
    returns for each radius the corresponding quantity
    of uniformly distributed angles on a cirle.
    """
    for i in range(len(radii)):
        radius = radii[i]
        number = numbers[i]
        for n in range(number):
            angle = (2*math.pi*n)/(number)
            yield angle, radius

def rtuniform(n=5,rmax=1,m=6):
    """
    For n evenly spaced radii between 0 and rmax, returns m * radius no.
    uniformly distributed angles on a circle.
    """
    radii = []
    numbers = []
    for i in range(n+1):
        radius = (i*rmax)/float(n)
        #print radius
        if i == 0:
            number = 1
        else:
            number = m*i
        radii.append(radius)
        numbers.append(number)
    return rtpairs(radii, numbers)


for t,r in rtuniform(n=5, rmax=3, m=6):
    plt.polar(t, r, 'bo')

plt.show()
