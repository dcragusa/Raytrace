# Raytrace

A second-year physics undergraduate project at Imperial College London: implementing a raytracer.

* import raytracer as rt

* To plot a ray-trace, use rt.focusplot()
  * The default values should be sufficient.
  * Comment out lens2 if only using one lens.

* To plot RMS size vs Airy Pattern size, use rt.spotvsdiffraction()

* To find the RMS at the focus, use rt.rootmeansquare()
  * You can specify the output plane to calculate the RMS at with kwarg output.

* To run the optimiser, use rt.returnoptimised()
  * Modify kwarg x0 for different initial curvatures.

* To plot a rainbow ray-trace, use rt.rainbowplot()
  * The default values should be sufficient.

* To plot a rainbow through the prism, use rt.prismplot()
  * The default values should be sufficient.

* You can modify the lenses and prism values at lines 438-446.
