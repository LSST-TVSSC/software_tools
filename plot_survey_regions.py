import numpy as np
from astropy.coordinates import Angle
from regions import PixCoord, RectanglePixelRegion, CirclePixelRegion
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)

# Street, also Lund, Strader, Poleski
plane1 = RectanglePixelRegion(center=PixCoord(x=42.5,y=0.0),
                                    width=85.0, height=20,
                                    angle=Angle(0.0,'deg'))
plane2 = RectanglePixelRegion(center=PixCoord(x=317.5,y=0.0),
                                    width=85.0, height=20,
                                    angle=Angle(0.0,'deg'))
LMC = RectanglePixelRegion(center=PixCoord(x=280.4652, y=-32.8884),
                                    width=(322.827/60), height=(274.770/60),
                                    angle=Angle(0.0,'deg'))
SMC = RectanglePixelRegion(center=PixCoord(x=302.8084, y=-44.3277),
                                    width=(158.113/60), height=(93.105/60),
                                    angle=Angle(0.0,'deg'))
GalBulge = CirclePixelRegion(center=PixCoord(x=33.2,y=-3.13),
                                    radius=1.75)
survey_regions = [plane1, plane2, LMC, SMC, GalBulge]

for reg in survey_regions:
    patch = reg.as_artist(facecolor='none', edgecolor='red', lw=2)
    ax.add_patch(patch)

# Gonzalez survey region, Galactic Plane, between −15◦ < l < +15◦ and−10◦ < b < +10◦
plane1 = RectanglePixelRegion(center=PixCoord(x=7.5,y=0.0),
                                    width=15.0, height=20,
                                    angle=Angle(0.0,'deg'))
plane2 = RectanglePixelRegion(center=PixCoord(x=352.5,y=0.0),
                                    width=15.0, height=20,
                                    angle=Angle(0.0,'deg'))
survey_regions = [plane1, plane2]

for reg in survey_regions:
    patch = reg.as_artist(facecolor='none', edgecolor='blue', lw=2)
    ax.add_patch(patch)


# Bono survey region, Shallow survey of 20.l.+20 deg and -15.b.+10 deg,
# deep survey of 20.l.+20 deg and -3.b.+3 deg
shallow1 = RectanglePixelRegion(center=PixCoord(x=10.0,y=-2.5),
                                    width=10.0, height=25.0,
                                    angle=Angle(0.0,'deg'))
shallow2 = RectanglePixelRegion(center=PixCoord(x=350.0,y=-2.5),
                                    width=10.0, height=25.0,
                                    angle=Angle(0.0,'deg'))

deep1 = RectanglePixelRegion(center=PixCoord(x=10.0,y=0.0),
                                    width=10.0, height=6.0,
                                    angle=Angle(0.0,'deg'))
deep2 = RectanglePixelRegion(center=PixCoord(x=350.0,y=0.0),
                                    width=10.0, height=6.0,
                                    angle=Angle(0.0,'deg'))
for reg in [shallow1, shallow2]:
    patch = reg.as_artist(facecolor='none', edgecolor='purple', lw=2, alpha=0.3)
    ax.add_patch(patch)

for reg in [deep1, deep2]:
    patch = reg.as_artist(facecolor='none', edgecolor='purple', lw=2)
    ax.add_patch(patch)

# Clementini
M54 = CirclePixelRegion(center=PixCoord(x=5.60703,	y=-14.08715),
                                    radius=1.75)
Sculptor = CirclePixelRegion(center=PixCoord(x=287.5334, y=-83.1568),
                                    radius=1.75)
Carina = CirclePixelRegion(center=PixCoord(x=260.1124, y=-22.2235),
                                    radius=1.75)
Fornax = CirclePixelRegion(center=PixCoord(x=237.1038, y=-65.6515),
                                    radius=1.75)
Phoenix = CirclePixelRegion(center=PixCoord(x=272.1591, y=-68.9494),
                                    radius=1.75)
Antlia2 = CirclePixelRegion(center=PixCoord(x=264.8955, y=11.2479),
                                    radius=1.75)
survey_regions = [M54, Sculptor, Carina, Fornax, Phoenix, Antlia2]

for reg in survey_regions:
    patch = reg.as_artist(facecolor='none', edgecolor='green', lw=2)
    ax.add_patch(patch)

# Bonito
EtaCarina = CirclePixelRegion(center=PixCoord(x=287.5967884538, y=-00.6295111793),
                                    radius=1.75)
OrionNebula = CirclePixelRegion(center=PixCoord(x=209.0137, y=-19.3816),
                                    radius=1.75)
NGC2264 = CirclePixelRegion(center=PixCoord(x=202.9358, y=+02.1957),
                                    radius=1.75)
NGC6530 = CirclePixelRegion(center=PixCoord(x=6.0828, y=-01.3313),
                                    radius=1.75)
NGC6611 = CirclePixelRegion(center=PixCoord(x=16.9540, y=0.7934),
                                    radius=1.75)
# Components of the Gould belt structure?
survey_regions = [EtaCarina, OrionNebula, NGC2264, NGC6530, NGC6611]
for reg in survey_regions:
    patch = reg.as_artist(facecolor='none', edgecolor='orange', lw=2)
    ax.add_patch(patch)


plt.xlim(0.0, 360.0)
plt.ylim(-90.0, 90.0)
plt.xlabel('l [deg]')
plt.ylabel('b [deg]')
ax.set_aspect('equal')

plt.show()
