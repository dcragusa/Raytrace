"""
A script for investigating optical ray-tracing.
"""

import numpy as np
from numpy import dot as dotpr
from numpy.linalg import norm as mag
import matplotlib.pyplot as plt
import genpolar
from random import choice
from scipy.optimize import fmin_tnc as optimise


class Ray:
    """
    A class representing an optical ray. Accepts two
    lists of size 3, representing position and direction.
    The wavelength is quoted in micrometers.
    """
    def __init__(self, p, k, wv=0.588):
        self.__check(p,k)
        inp = np.array(p)
        ink = np.array(k)
        innormk = ink / mag(ink) # normalised direction vector
        self._log = [[inp, innormk]]
        self._wv = wv
        self._terminated = False

    def __check(self, p, k):
        "Sanity test for inputs."
        if type(p) != list:
            raise Exception("Point is not a list")
        if len(p) != 3:
            raise Exception("Point is not 3D")
        if k is None:
            pass
        else:
            if type(k) != list:
                raise Exception("Direction is not a list")
            if len(k) != 3:
                raise Exception("Direction is not 3D")

    def p(self):
        return self._log[-1][0]

    def k(self):
        return self._log[-1][1]

    def append(self, p, k):
        "Adds a new point and direction to the log."
        self.__check(p,k)
        inp = np.array(p)
        if k is None:
            self._log.append([inp, None])
        else:
            ink = np.array(k)
            innormk = ink / mag(ink)
            self._log.append([inp,innormk])

    def vertices(self):
        "Returns the path of the ray as a list of 3D points."
        output = []
        for i in self._log:
            output.append(i[0])
        return output


class OpticalElement():
    "A convenience class containing intercept functions for spherical elements and planes."
    def zerocurveintercept(self, ray):
        l = (self._z0 - ray.p()[2]) / ray.k()[2]
        if l > 0:
            q = ray.p() + l*ray.k()
            return q
        else:
            return None

    def intercept(self, ray):
        if self._curvature == 0:
            return self.zerocurveintercept(ray)
        r = ray.p() - self._origin
        k = ray.k()
        unitk = k / mag(k)
        rdotk = dotpr(r, unitk)
        if (rdotk)**2-((mag(r))**2-self._R**2) < 0:
            return None
        root = np.sqrt((rdotk)**2-((mag(r))**2-self._R**2))
        toreturn = []
        plusl = -(rdotk) + root
        if plusl > 0: # need forward direction of ray
            plusq = ray.p() + (plusl * unitk)
            plusqzdist = mag(plusq[0:2])-self._apradius
            if plusqzdist <= 0: # if intercept is within aperture radius
                toreturn.append(plusq)
        minusl = -(rdotk) - root
        if minusl > 0:
            minusq = ray.p() + (minusl * unitk)
            minusqzdist = mag(minusq[0:2])-self._apradius
            if minusqzdist <= 0:
                toreturn.append(minusq)
        if len(toreturn) == 0:
            return None
        elif len(toreturn) == 1:
            return toreturn[0]
        elif len(toreturn) == 2:
            if self._curvature > 0:
                if plusl <= minusl: # for +ve curv. we require smaller l
                    return plusq
                else:
                    return minusq
            else:
                if plusl >= minusl: # for -ve curv. we require larger l
                    return plusq
                else:
                    return minusq

class SphericalRefraction(OpticalElement):
    """
    A spherical surface for refraction, with origin on the z optical axis. Called with the intercept
    of the surface on the z axis, the curvature of the surface, two refractive indices (ray incoming
    side first) and the aperture radius of the surface. Zero curvature represents a plane surface.
    """
    def __init__(self, intercept, curvature, n1=1, n2=1.5168, apradius=5):
        self._z0 = intercept
        self._curvature = curvature
        self._n1 = n1
        self._n2 = n2
        self._apradius = apradius
        try:
            self._R = 1/self._curvature
        except ZeroDivisionError: # plane surface
            self._R = 0
        self._origin = np.array([0,0,self._z0+self._R])

    def snellrefract(self, incident, normal, n1, n2):
        ndotk = dotpr(normal, incident)
        if ndotk < 0: # we require positive n dot k
            ndotk = dotpr(-normal, incident)
        ratio = n1 / float(n2)
        sinincident = np.sqrt(1-(ndotk**2))
        if sinincident > n2/float(n1): # discard total internal reflection
            return None
        refracted = ratio*incident + (ratio*ndotk - np.sqrt(1-((ratio**2)*(1-ndotk**2))))*normal
        normrefracted = refracted / mag(refracted)
        return normrefracted

    def propagate_ray(self, ray):
        if ray._terminated: # do not propagate if previously terminated
            return None
        q = self.intercept(ray)
        if q is None:
            ray._terminated = True
            return None
        unitk = ray.k() / mag(ray.k())
        if self._curvature == 0: # set normal for plane surface
            normal = np.array([0,0,-1])
        elif self._curvature < 0:        # we require the normal to be
            normal = self._origin - q    # opposite the ray direction
        else:
            normal = q - self._origin
        unitnormal = normal / mag(normal)
        newdirection = self.snellrefract(unitk, unitnormal, self._n1, self._n2)
        if newdirection is None:
            ray._terminated = True
            return None
        ray.append(q.tolist(), newdirection.tolist())

    def disperse_ray(self, ray):
        if ray._terminated:
            return None
        q = self.intercept(ray)
        if q is None:
            ray._terminated = True
            return None
        unitk = ray.k() / mag(ray.k())
        if self._curvature == 0:
            normal = np.array([0,0,-1])
        elif self._curvature < 0:
            normal = self._origin - q
        else:
            normal = q - self._origin
        unitnormal = normal / mag(normal)
        wv = ray._wv
        n2 = np.sqrt(1 + (1.03961212*wv**2)/(wv**2-0.00600069867) + (0.231792344*wv**2)/(wv**2-0.0200179144)
                       + (1.01046945*wv**2)/(wv**2-103.560653))
        newdirection = self.snellrefract(unitk, unitnormal, self._n1, n2)
        if newdirection is None:
            ray._terminated = True
            return None
        ray.append(q.tolist(), newdirection.tolist())

class SphericalReflection(OpticalElement):
    """
    A spherical surface for reflection, with origin on the z optical axis. Called
    with the intercept of the surface on the z axis, the curvature of the surface,
    and the aperture radius of the surface. Zero curvature represents a plane surface.
    """
    def __init__(self, intercept, curvature, apradius):
        self._z0 = intercept
        self._curvature = curvature
        self._apradius = apradius
        try:
            self._R = 1/self._curvature
        except ZeroDivisionError:
            self._R = 0
        self._origin = np.array([0,0,self._z0+self._R])

    def reflect(self, incident, normal):
        ndotk = dotpr(normal, incident)
        reflected = incident - 2*(ndotk)*normal
        normreflected = reflected / mag(reflected)
        return normreflected

    def propagate_ray(self, ray):
        if ray._terminated:
            return None
        q = self.intercept(ray)
        if q is None:
            ray._terminated = True
            return None
        unitk = ray.k() / mag(ray.k())
        if self._curvature == 0:
            normal = np.array([0,0,-1])
        elif self._curvature < 0:
            normal = self._origin - q
        else:
            normal = q - self._origin
        unitnormal = normal / mag(normal)
        newdirection = self.reflect(unitk, unitnormal)
        if newdirection is None:
            ray._terminated = True
            return None
        ray.append(q.tolist(), newdirection.tolist())

class Prism(OpticalElement):
    def __init__(self, order):
        self._order = order
        if self._order == '1':
            self._normal = np.array([2,0,-1])
        elif self._order == '2':
            self._normal = np.array([2,0,-1])

    def intercept(self, ray):
        if self._order == '1':
            l = (100-ray.p()[0] - ray.p()[2]) / ray.k()[2]
        if self._order == '2':
            l = (100+ray.p()[0] - ray.p()[2]) / ray.k()[2]
        if l > 0:
            q = ray.p() + l*ray.k()
            return q
        else:
            return None

    def snellrefract(self, incident, normal, n1, n2):
        ndotk = dotpr(normal, incident)
        if ndotk < 0: # we require positive n dot k
            ndotk = dotpr(-normal, incident)
        ratio = n1 / float(n2)
        sinincident = np.sqrt(1-(ndotk**2))
        if sinincident > n2/float(n1): # discard total internal reflection
            return None
        refracted = ratio*incident + (ratio*ndotk - np.sqrt(1-((ratio**2)*(1-ndotk**2))))*normal
        normrefracted = refracted / mag(refracted)
        return normrefracted

    def disperse_ray(self, ray):
        if ray._terminated:
            return None
        q = self.intercept(ray)
        if q is None:
            ray._terminated = True
            return None
        unitk = ray.k() / mag(ray.k())
        normal = self._normal
        unitnormal = normal / mag(normal)
        wv = ray._wv
        n2 = np.sqrt(1 + (1.03961212*wv**2)/(wv**2-0.00600069867) + (0.231792344*wv**2)/(wv**2-0.0200179144)
                       + (1.01046945*wv**2)/(wv**2-103.560653))
        newdirection = self.snellrefract(unitk, unitnormal, 1, n2)
        if newdirection is None:
            ray._terminated = True
            return None
        ray.append(q.tolist(), newdirection.tolist())

class OutputPlane(OpticalElement):
    """
    An infinite xy plane, called with the z-axis
    coord. Its purpose is to terminate the rays.
    """
    def __init__(self, z):
        self._z0 = z
        self._curvature = 0

    def propagate_ray(self, ray):
        if ray._terminated:
            return None
        q = self.intercept(ray)
        ray.append(q.tolist(), None)


class RayData:
    "A container class for rays. Holds a convenience function for propagation."
    def propagate(self, surface):
        for ray in self._rays:
            surface.propagate_ray(ray)

class ParaxialFocus(RayData):
    "Passes one ray 0.1mm from the optical axis to find the paraxial focus of the system."
    def __init__(self):
        self._rays = [Ray([0.1, 0, 0], [0,0,1])]

    def focus(self):
        for ray in self._rays:
            l = (0 - ray.p()[0]) / ray.k()[0]
            q = ray.p() + l*ray.k()
            return q[2]

class RayBundle(RayData):
    """
    A class representing a bundle of rays. Initialised with no. of rings, max radius,
    and a scale factor (all of which are passed to genpolar), plus an xy offset and
    a direction vector that are applied to each coordinate to make rays.
    """
    def __init__(self, nofradii=5, maxradius=5, factor=6, center=[0,0], directions=[0,0,1]):
        self._rays = []
        for t,r in genpolar.rtuniform(n=nofradii, rmax=maxradius, m=factor):
            x = r*np.cos(t)
            y = r*np.sin(t)
            ray = Ray([center[0]+x, center[1]+y, 0], directions)
            self._rays.append(ray)

    def plot(self):
        """
        Plots the bundle of rays. This is called once all propagations have
        been made. The first subplot is an xz section of the raytrace through
        the system, and the second is an xy spot diagram at the paraxial focus.
        """
        spotx = []
        spoty = []
        for ray in self._rays:
            if ray._terminated:
                pass
            else:
                listvertices = ray.vertices()
                listx = []
                listy = []
                for vertex in listvertices:
                    listx.append(vertex[2])
                    listy.append(vertex[0])
                plt.subplot(1, 2, 1)
                plt.plot(listx, listy, color='blue')
                plt.title('Raytrace in the xz plane')
                plt.xlabel('z (mm)')
                plt.ylabel('x (mm)')

                spotx.append(ray.p()[0])
                spoty.append(ray.p()[1])

        plt.subplot(1, 2, 2)
        plt.scatter(spotx, spoty, color='blue')
        plt.axis([min(spotx)*1.1, max(spotx)*1.1, min(spoty)*1.1, max(spoty)*1.1])
        plt.title('Ray intercepts at the focus')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')

        plt.tight_layout()
        plt.show()

    def rms(self):
        "Returns the rms of the spot size (the geometrical focus) at the paraxial focus."
        listmags = []
        for ray in self._rays:
            mag = (ray.p()[0])**2 + (ray.p()[1])**2
            listmags.append(mag)
        return np.sqrt(np.mean(listmags))

class RainbowBundle(RayData):
    """
    A class representing a bundle of rays with random differing wavelengths. Initialised with no.
    of rings, max radius, and a scale factor (all of which are passed to genpolar), plus
    an xy offset and a direction vector that are applied to each coordinate to make rays.
    """
    def __init__(self, nofradii=5, maxradius=5, factor=6, center=[0,0], directions=[0,0,1]):
        wavelengths = [0.7,0.62,0.58,0.53,0.47,0.42,0.35]
        self._wvdict = {'0.7':'#FF0000','0.62':'#FF7F00','0.58':'#FFFF00','0.53':'#00FF00',
                        '0.47':'#0000FF','0.42':'#4B0082','0.35`':'#8F00FF'}
        self._rays = []
        for t,r in genpolar.rtuniform(n=nofradii, rmax=maxradius, m=factor):
            x = r*np.cos(t)
            y = r*np.sin(t)
            ray = Ray([center[0]+x, center[1]+y, 0], directions, choice(wavelengths))
            self._rays.append(ray)

    def disperse(self, surface):
        for ray in self._rays:
            surface.disperse_ray(ray)

    def plot(self):
        """
        Plots the bundle of rays. This is called once all propagations have
        been made. The first subplot is an xz section of the raytrace through
        the system, and the second is an xy spot diagram at the paraxial focus.
        """
        spotx = []
        spoty = []
        for ray in self._rays:
            if ray._terminated:
                pass
            else:
                listvertices = ray.vertices()
                listx = []
                listy = []
                for vertex in listvertices:
                    listx.append(vertex[2])
                    listy.append(vertex[0])
                plt.subplot(1, 2, 1)
                plt.plot(listx, listy, color=self._wvdict[str(ray._wv)])
                plt.title('Raytrace in the xz plane')
                plt.xlabel('z (mm)')
                plt.ylabel('x (mm)')

                spotx.append(ray.p()[0])
                spoty.append(ray.p()[1])

        plt.subplot(1, 2, 2)
        plt.scatter(spotx, spoty, color=self._wvdict[str(ray._wv)])
        plt.axis([min(spotx)*1.1, max(spotx)*1.1, min(spoty)*1.1, max(spoty)*1.1])
        plt.title('Ray intercepts at the focus')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')

        plt.show()


# rays = RayBundle()

# Refraction
lens = SphericalRefraction(100,0.02,1.0,1.5168,500)
lens2 = SphericalRefraction(105,-0.02,1.5168,1.0,500)

# Reflection
#lens = SphericalReflection(100,-0.1,500)

# Dispersion
prism = Prism('1')
prism2 = Prism('2')

# Wavelength
wavelength = 588e-6


def findfocus():
    "Find the paraxial focus of the system."
    rays = ParaxialFocus()
    rays.propagate(lens)
    rays.propagate(lens2)
    return rays.focus()

def sendbundle(output, rays):
    "Propagates the ray bundle through the system."
    outputplane = OutputPlane(output)
    rays.propagate(lens)
    #rays.propagate(lens2)
    rays.propagate(outputplane)
    return rays

def focusplot(output=findfocus(), raysin=RayBundle()):
    "Refer to RayBundle.plot()"
    rays = sendbundle(output, raysin)
    rays.plot()

def rootmeansquare(raysin=RayBundle(), output=findfocus()):
    "Returns the rms of the spot size (the geometrical focus) at the paraxial focus."
    output = findfocus()
    #output = 150
    rays = sendbundle(output, raysin)
    return rays.rms()

def spotvsdiffraction():
    """
    Plots the rms spot size at the paraxial focus (green) and the theoretical diffraction
    limit (red) of the system for ray bundle radii varying from 0.1mm to 5mm.
    """
    listx = []
    listyspot = []
    listydiff = []
    foclength = findfocus() - 102.5
    for i in range(2, 101):
        radius = i/float(20)
        rays = RayBundle(maxradius=radius)
        listx.append(radius)
        rms = rootmeansquare(rays)
        listyspot.append(rms)
        diffractionlimit = (wavelength*foclength)/(radius*2)
        listydiff.append(diffractionlimit)

    plt.plot(listx, listyspot, color='green', label='RMS of geometrical focus')
    plt.plot(listx, listydiff, color='red', label='Airy Pattern size')
    plt.title('RMS and Airy Pattern size vs ray bundle size')
    plt.xlabel('Ray bundle radius (mm)')
    plt.ylabel('mm')
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,y1,0.1))
    plt.legend()
    plt.show()

def optimiserms(curvs, output):
    "The function to be optimised. Takes the two curvatures and returns the rms at the output."
    raysin = RayBundle()
    global lens
    lens = SphericalRefraction(100,curvs[0],1.0,1.5168,30)
    global lens2
    lens2 = SphericalRefraction(105,curvs[1],1.5168,1.0,30)
    rays = sendbundle(output, raysin)
    return rays.rms()

def returnoptimised():
    "Returns the curvatures optimised to produce the smallest rms at the output, z=150mm in this case."
    output = 150
    optimised = optimise(func=optimiserms, x0=[0.005,-0.005], args=(output,), bounds=[(0,0.5),(-0.5,0)], maxfun=200, approx_grad=True)
    print optimised

def rainbowplot(output=400, rays=RainbowBundle(10,10,10)):
    "Propagates the rainbow bundle through the system taking account of dispersion."
    outputplane = OutputPlane(output)
    rays.disperse(lens)
    rays.disperse(lens2)
    rays.propagate(outputplane)
    rays.plot()

def prismplot(output=1000, rays=RainbowBundle(10,5,10, [15,0])):
    "Propagates the rainbow bundle through a prism taking account of dispersion."
    outputplane = OutputPlane(output)
    rays.disperse(prism)
    rays.disperse(prism2)
    rays.propagate(outputplane)
    rays.plot()
