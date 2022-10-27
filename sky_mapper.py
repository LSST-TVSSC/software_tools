from gammapy.maps import Map
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from regions import CircleSkyRegion, RectangleSkyRegion, EllipseAnnulusSkyRegion
from gammapy.maps import RegionGeom
import numpy as np

#position = SkyCoord(0.0, 5.0, frame='galactic', unit='deg')

# Create a WCS Map: binsz=pixel size in deg, skydir=coordinate of map centre,
#
#m_wcs = Map.create(binsz=0.1, map_type='wcs', width=10.0)
#m_wcs.set_by_coord(([-1, -1], [2, 4]), [0.5, 1.5])
#m_wcs.fill_by_coord( ([-1, -1], [2, 4]) )

# Create a HPX Map
#m_hpx = Map.create(binsz=0.1, map_type='hpx', skydir=position, width=10.0)

#m_wcs.write('test_file.fits', hdu='IMAGE')
#m_wcs.plot(cmap="inferno", add_cbar=True, stretch="sqrt")
#plt.show()

#m = Map.create(npix=100,binsz=3/100, skydir=(83.63, 22.01), frame='icrs')
#m.data = np.add(*np.indices((100, 100)))

# A circle centered in the Crab position
# RegionGeom(region, **kwargs)
#circle = RegionGeom.create("icrs;circle(83.63, 22.01, 0.5)")

# A box centered in the same position
#box = RegionGeom.create("icrs;box(83.63, 22.01, 1,2,45)")

# An ellipse in a different location
#ellipse = RegionGeom.create("icrs;ellipse(84.63, 21.01, 0.3,0.6,-45)")

# An annulus in a different location
#annulus = RegionGeom.create("icrs;annulus(82.8, 22.91, 0.1,0.3)")

#m.plot(add_cbar=True)

# Default plotting settings
#circle.plot_region()

# Different line styles, widths and colors
#box.plot_region(lw=2, linestyle='--', ec='k')
#ellipse.plot_region(lw=2, linestyle=':', ec='white')

# Filling the region with a color
#annulus.plot_region(lw=2, ec='purple', fc='purple')

#Galactic_Center = SkyCoord('17:45:40.04', '−29:00:28.1', frame="icrs", unit=(u.hourangle, u.deg))
Galactic_Center = SkyCoord(0.0, 0.0, frame="galactic", unit=(u.deg, u.deg))
m = Map.create(binsz=2.0, width=(180.0, 24.0*15.0), skydir=Galactic_Center, frame="galactic", proj="TAN")
#m.data.fill(0.0)
m.plot(add_cbar=True)

plane_survey_region = RectangleSkyRegion(center=SkyCoord(0.0,0.0, frame='galactic', unit=(u.deg, u.deg)),
                                    width=(2.0*85.0) * u.deg, height=20 * u.deg,
                                    angle=0.0*u.deg)
LMC_region = RectangleSkyRegion(center=SkyCoord(280.4652, -32.8884, frame='galactic', unit=(u.deg, u.deg)),
                                    width=(322.827/60) * u.deg, height=(274.770/60) * u.deg,
                                    angle=0.0*u.deg)
SMC_region =
plane_survey_box = RegionGeom.create(plane_survey_region)
LMC_box = RegionGeom.create(LMC_region)

plane_survey_box.plot_region(lw=2, ec='purple', fc='purple', alpha=0.2)
LMC_box.plot_region(lw=2, ec='purple', fc='purple')

plt.show()
