"""An exposure time calculator for LSST.  Uses GalSim to draw a galaxy with specified magnitude,
shape, etc, and then uses the same image as the optimal weight function.  Derived from D. Kirkby's
notes on deblending.
"""
# This is the original code found at https://github.com/jmeyers314/LSST_ETC/blob/master/lsstetc.py, and the author deserves the acnowledgement of this code.
# Here the code has been simply modified to match point sources (I put the half-light radius to 0"), and a simple source peak counts has been included.
# To do that:
# - sky brightness has been updated from http://www.lsst.org/files/docs/gee_137.28.pdf to https://smtn-002.lsst.io
# - photometric saturation limits has bee considered as in https://www.lsst.org/sites/default/files/docs/sciencebook/SB_3.pdf
#	u,g,r,i,z,Y = 14.7,15.7,15.8,15.8,15.3 and 13.9, respectively
# - 15-seconds zero points have been computed by considering 65,000 counts at the aforementioned saturation limits. Then, the peak counts are estimated by simply evaluating the flux scaled to the actual exposure time (visit_time)
# A warning "SATURATION" has been included at a conservative level of 63,000 counts
# This ETC has been tested and it is in very good agreement with the 5-sigma limits reported in https://smtn-002.lsst.io/, that is
# u 	23.42
# g 	24.77
# r 	24.34
# i 	23.89
# z 	23.33
# y 	22.42
#
# M. Dall'Ora, 11-09-2018 (mmddyyyy)

from __future__ import print_function

import numpy as np

import galsim

# Some constants
# --------------
#
# LSST effective area in meters^2
A = 319/9.6  # etendue / FoV.  I *think* this includes vignetting

# zeropoints from DK notes in photons per second per pixel
# should eventually compute these on the fly from filter throughput functions.
s0 = {'u': A*0.732,
      'g': A*2.124,
      'r': A*1.681,
      'i': A*1.249,
      'z': A*0.862,
      'Y': A*0.452}
# Sky brightnesses in AB mag / arcsec^2.
# stole these from http://www.lsst.org/files/docs/gee_137.28.pdf
# should eventually construct a sky SED (varies with the moon phase) and integrate to get these
#B = {'u': 22.8,
#     'g': 22.2,
#     'r': 21.3,
#     'i': 20.3,
#     'z': 19.1,
#     'Y': 18.1}

# Updated with https://smtn-002.lsst.io
B = {'u': 22.95,
     'g': 22.24,
     'r': 21.20,
     'i': 20.47,
     'z': 19.60,
     'Y': 18.63}

#for k in B:
#print(format(B['r']))    

# number of visits
# From LSST Science Book
#fiducial_nvisits = {'u': 56,
#                    'g': 80,
#                    'r': 180,
#                    'i': 180,
#                    'z': 164,
#                    'Y': 164}

# Setting a single visit
fiducial_nvisits = {'u': 1,
                    'g': 1,
                    'r': 1,
                    'i': 1,
                    'z': 1,
                    'Y': 1}


# exposure time per visit
visit_time = 15.
# Sky brightness per arcsec^2 per second
sbar = {}
for k in B:
    sbar[k] = s0[k] * 10**(-0.4*(B[k]-24.0))


# And some random numbers for drawing
bd = galsim.BaseDeviate(1)


class ETC(object):
    def __init__(self, band, pixel_scale=None, stamp_size=None, threshold=0.0,
                 nvisits=None):
        self.pixel_scale = pixel_scale
        self.stamp_size = stamp_size
        self.threshold = threshold
        self.band = band
        if nvisits is None:
            self.exptime = fiducial_nvisits[band] * visit_time
        else:
            self.exptime = nvisits * visit_time
        self.sky = sbar[band] * self.exptime * self.pixel_scale**2
        self.sigma_sky = np.sqrt(self.sky)
        self.s0 = s0[band]

    def draw(self, profile, mag, noise=False):
        img = galsim.ImageD(self.stamp_size, self.stamp_size, scale=self.pixel_scale)
        flux = self.s0 * 10**(-0.4*(mag - 24.0)) * self.exptime
        profile = profile.withFlux(flux)
        profile.drawImage(image=img)
        if noise:
            gd = galsim.GaussianNoise(bd, sigma=self.sigma_sky)
            img.addNoise(gd)
        return img

    def SNR(self, profile, mag):
        img = self.draw(profile, mag, noise=False)
        mask = img.array > (self.threshold * self.sigma_sky)
        imgsqr = img.array**2*mask
        signal = imgsqr.sum()
        noise = np.sqrt((imgsqr * self.sky).sum())
        return signal / noise

    def err(self, profile, mag):
        snr = self.SNR(profile, mag)
        return 2.5 / np.log(10) / snr

    def display(self, profile, mag, noise=True):
        img = self.draw(profile, mag, noise)
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        plt.imshow(img.array, cmap=cm.Greens)
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()

    # Filter
    parser.add_argument("--band", default='z',
                        help="band for simulation (Default 'i')")

    # PSF structural arguments
    PSF_profile = parser.add_mutually_exclusive_group()
    PSF_profile.add_argument("--kolmogorov", action="store_true",
                             help="Use Kolmogorov PSF (Default Gaussian)")
    PSF_profile.add_argument("--moffat", action="store_true",
                             help="Use Moffat PSF (Default Gaussian)")
    parser.add_argument("--PSF_beta", type=float, default=2.5,
                        help="Set beta parameter of Moffat profile PSF. (Default 2.5)")
    parser.add_argument("--PSF_FWHM", type=float, default=0.67,
                        help="Set FWHM of PSF in arcsec (Default 0.67).")
    parser.add_argument("--PSF_phi", type=float, default=0.0,
                        help="Set position angle of PSF in degrees (Default 0.0).")
    parser.add_argument("--PSF_ellip", type=float, default=0.0,
                        help="Set ellipticity of PSF (Default 0.0)")

    # Galaxy structural arguments
    parser.add_argument("-n", "--sersic_n", type=float, default=1.0,
                        help="Sersic index (Default 1.0)")
    parser.add_argument("--gal_ellip", type=float, default=0.0,
                        help="Set ellipticity of galaxy (Default 0.3)")
    parser.add_argument("--gal_phi", type=float, default=0.0,
                        help="Set position angle of galaxy in radians (Default 0.0)")
    parser.add_argument("--gal_HLR", type=float, default=0.0,
                        help="Set galaxy half-light-radius. (default 0.5 arcsec)")

    # Simulation input arguments
    parser.add_argument("--pixel_scale", type=float, default=0.2,
                        help="Set pixel scale in arcseconds (Default 0.2)")
    parser.add_argument("--stamp_size", type=int, default=31,
                        help="Set postage stamp size in pixels (Default 31)")

    # Magnitude!
    parser.add_argument("--mag", type=float, default=18.,
                        help="magnitude of galaxy")
    # threshold
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Threshold, in sigma-sky units, above which to include pixels")

    # Observation characteristics
    parser.add_argument("--nvisits", type=int, default=None)

    # draw the image!
    parser.add_argument("--display", action='store_true',
                        help="Display image used to compute SNR.")

    args = parser.parse_args()

    if args.kolmogorov:
        psf = galsim.Kolmogorov(fwhm=args.PSF_FWHM)
    elif args.moffat:
        psf = galsim.Moffat(fwhm=args.PSF_FWHM, beta=args.PSF_beta)
    else:
        psf = galsim.Gaussian(fwhm=args.PSF_FWHM)
    psf = psf.shear(e=args.PSF_ellip, beta=args.PSF_phi*galsim.radians)

    gal = galsim.Sersic(n=args.sersic_n, half_light_radius=args.gal_HLR)
    gal = gal.shear(e=args.gal_ellip, beta=args.gal_phi*galsim.radians)

    profile = galsim.Convolve(psf, gal)

    etc = ETC(args.band, pixel_scale=args.pixel_scale, stamp_size=args.stamp_size,
              threshold=args.threshold, nvisits=args.nvisits)

    # Estimated counts
    # These numbers are based on the saturation limits listed in
    # https://www.lsst.org/sites/default/files/docs/sciencebook/SB_3.pdf
    # The factor '15' comes from the 15-seconds exposure used in the document
    # The zero-points are therefore computed on a 15s exposure, and then scaled by the actual visit_time

       
    if format(args.band) == 'u':
       counts = 10**(0.4*(26.73-args.mag))*visit_time/15

    if format(args.band) == 'g':
       counts = 10**(0.4*(27.73-args.mag))*visit_time/15

    if format(args.band) == 'r':
       counts = 10**(0.4*(27.83-args.mag))*visit_time/15

    if format(args.band) == 'i':
       counts = 10**(0.4*(27.83-args.mag))*visit_time/15

    if format(args.band) == 'z':
       counts = 10**(0.4*(27.33-args.mag))*visit_time/15

    if format(args.band) == 'Y':
       counts = 10**(0.4*(25.93-args.mag))*visit_time/15

#AA = A*1.681
#counts = A*1.681*visit_time * 10**(-0.4*(args.mag-24.0))


    print()
    print("input")
    print("------")
    print("band: {}".format(args.band))
    print("magnitude: {}".format(args.mag))
    print("exposure time: {}".format(visit_time))
    print("nvisits: {}".format(fiducial_nvisits[args.band] if args.nvisits is None else args.nvisits))
    print()
    print("output")
    print("------")
    print("SNR: {}".format(etc.SNR(profile, args.mag)))
    print("mag err: {}".format(etc.err(profile, args.mag)))
    print("counts: {}".format(counts))
    if counts > 63000:
       print("SATURATION!!")

    if args.display:
        etc.display(profile, args.mag)
