from os import path
from sys import argv
import random
import photutils
import numpy as np
from scipy import signal
from astropy.io import fits
from astropy.table import Table, Column
import matplotlib.pyplot as plt
import healpy as hp
from sklearn.cluster import DBSCAN, MiniBatchKMeans, OPTICS, MeanShift, AffinityPropagation
from pylab import cm
import generate_sky_maps

# Default configuration:
NSIDE = 64
NPIX = hp.nside2npix(NSIDE)
PIXAREA = hp.nside2pixarea(NSIDE,degrees=True)

class RubinFOV():
    """
    Rubin Field Of View is a circular footprint 3.5 deg in diameter.
    This object holds a pixels attribute with the indices of selected pixels.
    DiameterFactor (default=1) allows the user to increase the effective size
    of the footprint to allow for dithering etc.
    """

    def __init__(self, pixScale, diameterFactor=1):
        self.diameter = 3.5*diameterFactor     # Degrees, circular footprint
        self.pixelScale = pixScale
        nPixelDiameter = int(round(self.diameter / pixScale,0))
        if nPixelDiameter % 2 == 0:
            nPixelDiameter += 1
        iMidPixel = int((nPixelDiameter-1)/2)
        self.nx = nPixelDiameter
        self.ny = nPixelDiameter

        self.footprint = np.zeros((self.ny,self.nx), dtype='int')
        for row in range(0,iMidPixel,1):
            if row == 0:
                idx = iMidPixel
            else:
                idx = range(iMidPixel-row,iMidPixel+1+row,1)
            self.footprint[row,idx] = 1
            self.footprint[self.ny-row-1,:] = self.footprint[row,:]
        self.footprint[iMidPixel,:] = 1


    def calcImagePixels(self,image,xc,yc,verbose=False):
        """
        If we place a FoV at a given x,y centroid on an image, return
        arrays of the x,y pixel positions of each pixel that lands within the
        image.
        """

        # Take the pixel boundary of the centroid position to ensure we are
        # dealing with whole pixels here
        xb = np.floor(xc)
        yb = np.floor(yc)
        if verbose: print('X boundary: '+str(xb)+', Y boundary: '+str(yb))

        # Calculate the half-width of the FoV footprint in pixels, ensuring
        # an odd number of pixels to maintain FOV shape
        xfov = round((self.nx/2.0),0)
        yfov = round((self.ny/2.0),0)
        if (xfov % 2) == 0:
            xfov -= 1
        if (yfov % 2) == 0:
            yfov -= 1
        if verbose: print('X FoV: '+str(xfov)+', Y FoV: '+str(yfov))

        # Determine the range of pixel values within the FoV
        xmin = max(0,(xb-xfov))
        xmax = min((xb+xfov+1),image.shape[1])
        ymin = max(0,(yb-yfov))
        ymax = min((yb+yfov+1),image.shape[0])
        X = np.arange(xmin,xmax,1)
        Y = np.arange(ymin,ymax,1)
        if verbose: print('X range: '+str(X)+', Y range: '+str(Y))

        # Generate 2D grid:
        (XX,YY) = np.meshgrid(X,Y)

        # Exclude those pixel locations that lie outside the image boundaries
        maskX = np.ones(XX.shape).astype('bool')
        maskX.fill(False)

        maskY = np.ones(YY.shape).astype('bool')
        maskY.fill(False)

        # Mask out the pixels not within the circular footprint:
        idx = np.where(self.footprint[0:len(Y),0:len(X)] == 0)
        maskX[idx] = True
        maskY[idx] = True

        # Create the masked pixel array
        # Note axis swap here
        pixels = ((YY[maskY == False]).astype('int'),
                  (XX[maskX == False]).astype('int'))

        return pixels

class PixelCluster():
    """Class of object describing a set of HEALpixels"""

    def __init__(self,xc=None, yc=None):
        self.x_center = xc
        self.y_center = yc
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None
        self.width = None
        self.pixels = np.array([])  # Two axis array, YY, XX
        self.priority = 0.0

    def calcCentroid(self):
        self.x_center = self.xmin + (self.xmax-self.xmin)/2.0
        self.y_center = self.ymin + (self.ymax-self.ymin)/2.0

    def calcClusterRanges(self):
        if len(self.pixels) > 0:
            # Maximum box size, including outer edges of FoV on all sides
            self.xmin = self.pixels[1].min()
            self.xmax = self.pixels[1].max()
            self.ymin = self.pixels[0].min()
            self.ymax = self.pixels[0].max()

            self.width = max( (self.xmax-self.xmin), (self.ymax-self.ymin) )
        else:
            self.xmin = None
            self.xmax = None
            self.ymin = None
            self.ymax = None
            self.width = None

    def calcSeparation(self,cluster2):
        # This needs updating to use the RA/Dec separation not cartesian
        dx = self.x_center - cluster2.x_center
        dy = self.y_center - cluster2.y_center
        return np.sqrt(dx*dx + dy*dy)

    def sumPixelPriority(self, priorityMap):
        self.priority = priorityMap[self.pixels].sum()

    def summary(self):
        return 'Cluster center ('+str(self.x_center)+\
                ', '+str(self.y_center)+\
                ') with ranges '+str(self.xmin)+':'+str(self.xmax)+\
                ', '+str(self.ymin)+':'+str(self.ymax)+\
                ' and '+str(len(self.pixels[0]))+' pixels: '+repr(self.pixels)

    def calcPixelArea(self,pixArea):
        return len(self.pixels[0])*pixArea

    def calcPixelPriorityStats(self, priorityMap):
        self.avgPriority = priorityMap[self.pixels].sum()/len(self.pixels[0])
        self.stdPriority = priorityMap[self.pixels].std()
        self.maxPriority = priorityMap[self.pixels].max()

    def zipPixels(self):
        """Method returns the pixel (x,y) positions as a list of tuples"""
        return zip(list(self.pixels[0]),list(self.pixels[1]))

def analyse_image(params):

    if not path.isdir(params['input_dir']):
        raise IOError('Cannot find input directory '+params['input_dir'])

    # Load image data from galactic science priority map converted to Mollweide
    # image format, as well as the metadata about each pixel in HEALpixel space
    (priorityMap, pixelMetaData) = load_map_image_data(params['image_path'])

    # Estimate pixel area:
    pixHeight = 180.0 / priorityMap.shape[0]
    pixWidth = 360.0 / priorityMap.shape[1]
    pixArea = pixHeight * pixWidth

    # Define the Rubin field of view array:
    FoV = RubinFOV(pixScale=pixWidth)

    # Explore range of priority thresholds:
    priorityRange = np.arange(params['priority_min'],
                                params['priority_max'],
                                    params['priority_step'])
    output = open(path.join(params['output_dir'],'survey_area_priority_data.dat'), 'w')
    output.write('#PriorityThreshold    Area in survey   Sum(priority)\n')
    for priorityThreshold in priorityRange:
        # Object detection on the original image returns a lot of close-neighbour hits
        # Trying convolution with the Rubin footprint
        # This works, in that it reduces the number of multiple detections within
        # a single feature in the map.  The downside is that it modifies the array
        # indexing, which makes it much harder to backout later the HEALpixel IDs
        # (and hence RA, Dec) of the regions.
        #smoothedImage = signal.convolve(image, FoV)
        #print('Dimensions of smoothed image: ',smoothedImage.shape)

        # Background threshold for detection in this image represents
        # a minimum acceptable priority value:
        #priorityThreshold = params['priorityThreshold']
        avgPixelThreshold = priorityThreshold

        # Fractional Increase in the summed pixel priority required to accept a
        # supercluster, relative to the sum of the pixels in the
        # component clusters:
        fracIncrease = 0.0

        # Separation threhold within which clusters may be merged:
        separationThreshold = FoV.nx*4.0

        # Perform object detection on smoothed 2D priority map
        objectTable = photutils.detection.find_peaks(priorityMap,
                                                     priorityThreshold,
                                                     footprint=FoV.footprint)
        print(objectTable)
        print(objectTable['peak_value'].min())

        # Plot detected objects
        fig = plt.figure(1,(10,10))
        imgplot = plt.imshow(priorityMap)
        plt.plot(objectTable['x_peak'], objectTable['y_peak'], 'b+')
        plt.title('Galactic science priority map with detected peaks')
        plt.savefig(path.join(params['output_dir'],
                    'map_detected_priority_peaks_p'+str(round(priorityThreshold,2))
                    +'_a'+str(round(avgPixelThreshold,2))+'_'+params['suffix']+'.png'))
        plt.close(1)

        # Next step is find clusters of detections in the object table,
        # which represent strong peaks in the priority map.
        # Note that map projection is neglected here as we are working in
        # pixel space
        #peaksTable = findClusters(objectTable)
        peaksTable = initClusters(objectTable)
        print(peaksTable)

        # Plot location of clusters
        fig = plt.figure(2,(10,10))
        imgplot = plt.imshow(priorityMap)
        plt.plot(peaksTable['x_center'], peaksTable['y_center'], 'rD')
        plt.title('Galactic science priority map showing clusters of peaks')
        plt.savefig(path.join(params['output_dir'],
                    'map_priority_peaks_p'+str(round(priorityThreshold,2))
                    +'_a'+str(round(avgPixelThreshold,2))+'_'+params['suffix']+'.png'))
        plt.close(2)

        # Build a revised map by placing a RubinFOV at each of the identified
        # clusters
        (mapImage, peaksPriorityMap) = buildMapFromPeaks(peaksTable,
                                                          priorityMap,
                                                          FoV)

        # Plot map of Rubin FoV on cluster locations
        fig = plt.figure(3,(10,10))
        imgplot = plt.imshow(peaksPriorityMap)
        plt.title('Galactic science priority map showing peaks')
        plt.savefig(path.join(params['output_dir'],
                    'map_peaks_FoV_p'+str(round(priorityThreshold,2))
                    +'_a'+str(round(avgPixelThreshold,2))+'_'+params['suffix']+'.png'))
        plt.close(3)

        # Build list of pixelClusters.
        # Initially, the clusters are defined to be those within the RubinFoV
        # of a peak in the priority map.
        clusters = initPixelClustersList(peaksTable,FoV,priorityMap,priorityThreshold)
        file_name = path.join(params['output_dir'],
                    'map_init_cluster_pixels_p'+str(round(priorityThreshold,2))
                    +'_a'+str(round(avgPixelThreshold,2))+'_'+params['suffix']+'.png')
        plotMapOfClusters(clusters,priorityMap,file_name)

        # Merge neighboring clusters with overlapping pixel maps:
        clusters = mergeOverlappingClusters(clusters, priorityMap,
                                            avgPixelThreshold, fracIncrease)
        file_name = path.join(params['output_dir'],
                    'map_merged_overlap_clusters_p'
                    +str(round(priorityThreshold,2))
                    +'_a'+str(round(avgPixelThreshold,2))+'_'+params['suffix']+'.png')
        plotMapOfClusters(clusters,priorityMap,file_name)

        # Iterative loop
        max_iter = 1
        it = 1
        threshold = priorityThreshold
        while it <= max_iter:
            print('Iteration '+str(it))
            # Identify pixelClusters in close proximity, and test if superclusters
            # can be formed.
            clusters = identifySuperClusters(clusters, priorityMap,
                                                separationThreshold,
                                                priorityThreshold, fracIncrease,
                                                FoV)

            # Filter out isolated, low-priority clusters:
            #threshold += priorityThreshold*0.1*it
            #clusters = filterClusters(clusters, avgPixelThreshold, priorityMap)

            # Summarize the characteristics of the final survey region:
            plotClustersMap(params, clusters, priorityMap,it)
            calcAreaOfClusters(clusters,pixArea)
            calcPriorityofClusters(clusters, priorityMap)

            it += 1
        file_name = path.join(params['output_dir'],
                    'map_final_cluster_pixels_p'+str(round(priorityThreshold,2))
                    +'_a'+str(round(avgPixelThreshold,2))+'_'+params['suffix']+'.png')
        plotMapOfClusters(clusters,priorityMap,file_name)

        # Output the aggregated map as a HEALpixel map:
        selectedPixels = outputClusterMap(params, clusters, pixelMetaData,
                                            priorityThreshold, avgPixelThreshold)

        # Output the map data tables per filter:
        outputClusterMapDataTables(params, selectedPixels,
                                            priorityThreshold, avgPixelThreshold)

        # Summarize total area selected:
        idx = np.where(selectedPixels > 0)[0]
        surveyArea = PIXAREA * len(idx)
        totalPriority = selectedPixels[idx].sum()
        print('Total selected survey area: '+str(round(surveyArea,2))+'sq. deg')
        print('Total priority with the survey area: '+str(round(totalPriority)))
        output.write(str(priorityThreshold)+' '+str(surveyArea)+' '+str(totalPriority)+'\n')

    output.close()

def load_map_image_data(image_path):
    """Function to load the multi-extension FITS file produced by
    project_galmap_mollweide.py"""

    hdul = fits.open(image_path)
    rawImage = hdul[0].data
    idx = np.where(rawImage != -np.inf)
    print('Number of valid map pixels: '+str(len(idx[0])))
    priorityMap = np.zeros(rawImage.shape)
    priorityMap[idx] = rawImage[idx]
    print('Dimensions of input image: ',priorityMap.shape)
    table_data = []
    hdul_table = hdul[1].data
    for fits_column in hdul[1].columns:
        if 'J' in fits_column.format:
            type = 'int'
        elif 'D' in fits_column.format:
            type = 'float'
        elif 'A' in fits_column.format:
            type = 'str'
        else:
            type = 'str'
        table_data.append(Column(name=fits_column.name,
                                data=hdul_table[fits_column.name],
                                dtype=type))
    pixelMetaData = Table(table_data)

    return priorityMap, pixelMetaData

def initClusters(objectTable):
    """
    Function to initialize a cluster of peaks at each centroid peak detected.
    """
    print('Initializing cluster list for all '+str(len(objectTable))+' detected peaks')

    return Table([Column(name='x_center', data=objectTable['x_peak'].data),
                  Column(name='y_center', data=objectTable['y_peak'].data)])

def findClusters(objectTable):
    """
    Object detection functions sometimes produce multiple hits on the same object,
    creating several points within the same FoV.  This function aims to
    coalse these points to a single object.
    """
    X = objectTable['x_peak'].data
    Y = objectTable['y_peak'].data

    separations = np.zeros((len(objectTable),len(objectTable)))
    for j in range(0,len(objectTable),1):
        separations[:,j] = np.sqrt((X-X[j])**2 + (Y-Y[j])**2)

        # In this map, the X direction
    # Performs reasonably well, but misses some of the smaller regions
    #model = AffinityPropagation(damping=0.7)

    # Poor clustering, finds 3 large sections
    #model = MeanShift()

    model = MiniBatchKMeans(n_clusters=50)

    labels = model.fit_predict(separations)

    clusters = np.unique(labels)
    print('Found '+str(len(clusters))+' clusters')

    centroids = np.zeros((len(clusters),2))

    for i,cluster_id in enumerate(clusters):
        idx = np.where(labels == cluster_id)[0]
        centroids[i,0] = np.median(X[idx])
        centroids[i,1] = np.median(Y[idx])

    return Table([Column(name='x_center', data=centroids[:,0]),
                  Column(name='y_center', data=centroids[:,1])])

def buildMapFromPeaks(peaksTable,priorityMap,RubinFoV):
    """
    Function to build a priority map based on the table of clusters,
    by placing a RubinFoV at the centroid of each cluster
    """

    mapImage = np.zeros(priorityMap.shape)
    clusterPriorityMap = np.zeros(priorityMap.shape)
    for i in range(0,len(peaksTable),1):
        pixels = RubinFoV.calcImagePixels(priorityMap,
                                          peaksTable['x_center'][i],
                                          peaksTable['y_center'][i],
                                          )
        clusterPriorityMap[pixels] = priorityMap[pixels]
        mapImage[pixels] = 1.0

    return mapImage, clusterPriorityMap

def plotMapOfClusters(clusters,priorityMap,file_name):
    """
    Function to build a priority map based on the table of clusters,
    by placing a RubinFoV at the centroid of each cluster
    """

    mapImage = np.zeros(priorityMap.shape)
    xclusters = []
    yclusters = []
    for i,member in clusters.items():
        mapImage[member.pixels] = priorityMap[member.pixels]
        xclusters.append(member.x_center)
        yclusters.append(member.y_center)

    fig = plt.figure(6,(10,10))
    imgplot = plt.imshow(mapImage)
    plt.plot(xclusters, yclusters, 'r.')
    plt.title('Pixels included in clusters')
    plt.savefig(file_name)
    plt.close(6)

def initPixelClustersList(peaksTable,FoV,priorityMap,priorityThreshold):
    """
    Function to create a list of pixelClusters from the table of peaks in the
    priority map.  To start with, a RubinFoV is placed at the top of each
    detected peak to initialize a pixelCluster at that location.
    """

    # Produce a 2D pixel grid for the whole image
    xrange = range(0,priorityMap.shape[1],1)
    yrange = range(0,priorityMap.shape[0],1)
    (XX,YY) = np.meshgrid(xrange,yrange)
    radius = FoV.nx/2.0

    clusters = {}
    for i in range(0,len(peaksTable),1):
        x = peaksTable['x_center'][i]
        y = peaksTable['y_center'][i]
        cluster = PixelCluster(x,y)
        #cluster.pixels = FoV.calcImagePixels(priorityMap,x,y)

        xc = int(round((cluster.x_center),0))
        yc = int(round((cluster.y_center),0))
        separations = np.sqrt((XX-xc)**2 + (YY-yc)**2)
        mask = separations <= radius
        while (priorityMap[mask] >= priorityThreshold).all():
            radius += FoV.nx
            separations = np.sqrt((XX-xc)**2 + (YY-yc)**2)
            mask = separations <= radius

        aperturePixels = priorityMap[mask]
        idx = np.where(aperturePixels > priorityThreshold)[0]
        if len(idx) > 0:
            cluster.pixels = (YY[mask][idx],XX[mask][idx])

            cluster.calcClusterRanges()
            clusters[i] = cluster

    print('Created list of '+str(len(clusters))+' PixelClusters')

    return clusters

def mergeOverlappingClusters(clusters, priorityMap, priorityThreshold,
                            fracIncrease, verbose=False):
    clusterList = list(clusters.values())
    mergedClusters = True

    while mergedClusters:
        mergedClusters = False
        newClusterList = []
        for i in range(0,len(clusterList)-1,2):
            cluster1 = clusterList[i]
            cluster2 = clusterList[i+1]
            pixels1 = cluster1.zipPixels()
            pixels2 = cluster2.zipPixels()
            neighbors = {0: cluster1, 1: cluster2}
            overlap = set(pixels1).intersection(set(pixels2))
            if verbose: print('Evaluating clusters at ',
                        cluster1.x_center,cluster1.y_center,
                        cluster2.x_center,cluster2.y_center)
            if len(overlap) > 0:
                sc = mergeClustersEllipse(neighbors, priorityMap, priorityThreshold)
                accept_sc = reviewSuperCluster(sc, neighbors,
                                            priorityMap, fracIncrease)
                if verbose: print('Merge accepted? ',accept_sc)
                if accept_sc:
                    newClusterList.append(sc)
                    mergedClusters = True
                    if verbose: print('Added supercluster to newClusters list: ',
                                        newClusterList)
                else:
                    newClusterList += list(neighbors.values())
                    if verbose: print('Added component clusters to newClusters list: ',
                                list(neighbors.values()),newClusterList)
            else:
                newClusterList += list(neighbors.values())
                if verbose: print('No overlap, added component clusters: ',
                                list(neighbors.values()),newClusterList)
        if len(clusterList)%2 > 0:
            newClusterList.append(clusterList[-1])

        clusterList = newClusterList
        #random.shuffle(clusterList)

    # Re-index the revised dictionary of clusters
    newClusters = {}
    for i in range(0,len(newClusterList),1):
        newClusters[i] = newClusterList[i]

    print('Merged overlapping clusters.  Started with '+str(len(clusters))+
            ' clusters and finished with '+str(len(newClusters)))

    return newClusters

def identifySuperClusters(clusters, priorityMap, priorityThreshold,
                            separationThreshold, fracIncrease, FoV):
    """
    Function to test whether pixelClusters in proximity can be merged to form
    superclusters.
    """
    newClusterList = []
    mergedClusters = []

    for i,cluster in clusters.items():
        print('Searching for neighbors around '+\
                str(cluster.x_center)+', '+str(cluster.y_center))

        # Find the closest neighbour clusters; i.e. those with separations
        # smaller than their combined widths are candidates for merging
        neighbors = {}
        neighbors[i] = cluster
        for j, pairCluster in clusters.items():
            if j != i and j not in mergedClusters:
                separation = cluster.calcSeparation(pairCluster)
                if separation <= separationThreshold:
                    neighbors[j] = pairCluster
                    print(' -> Found neighbor at '+\
                            str(pairCluster.x_center)+', '+str(pairCluster.y_center)+\
                            ' sep='+str(separation)+' threshold='+str(separationThreshold))

        # Form a candidate supercluster from the list of component clusters:
        sc = mergeClustersEllipse(neighbors, priorityMap, priorityThreshold)
        print('Formed a candidate supercluster at '+str(sc.x_center)+\
                                                ', '+str(sc.y_center),\
              ' with ranges '+str(sc.xmin)+':'+str(sc.xmax)+\
              ', '+str(sc.ymin)+':'+str(sc.ymax)+\
              ' with '+str(len(sc.pixels[0]))+' pixels')

        # Test whether this supercluster meets the criteria for acceptance
        accept_sc = reviewSuperCluster(sc, neighbors,
                                    priorityMap, fracIncrease)
        print('Decision to accept SC: '+repr(accept_sc))

        if accept_sc:
            newClusterList.append(sc)
            for j,member in neighbors.items():
                mergedClusters.append(j)
        else:
            for j,member in neighbors.items():
                newClusterList.append(member)
            print('Retained component clusters')

    # Re-index the revised dictionary of clusters
    newClusters = {}
    for i in range(0,len(newClusterList),1):
        newClusters[i] = newClusterList[i]

    print(str(len(newClusters))+' remain after forming superclusters')

    return newClusters

def mergeClustersEllipse(clusters, priorityMap, priorityThreshold, verbose=False):
    """Function to merge clusters in a neighboring set as an elliptical region.
    """

    sc = PixelCluster()
    xpixels = np.array([],dtype='int')
    ypixels = np.array([],dtype='int')
    for i,member in clusters.items():
        xpixels = np.concatenate((xpixels, member.pixels[1]))
        ypixels = np.concatenate((ypixels, member.pixels[0]))
        if verbose: print('MERGE> sc concat pixels from members: ',member.pixels)
    sc.pixels = (np.array(ypixels),np.array(xpixels))
    if verbose: print('MERGE> sc concat pixels: ',sc.pixels)

    # Use the combined pixel arrays to calculate a rectangular bounding box
    sc.calcClusterRanges()
    sc.calcCentroid()
    if verbose: print('MERGE> sc init: ',sc.summary())

    # Draw an ellipse around both clusters and select the pixels within
    # the ellipse
    # Origin of the ellipse is the centroid of the supercluster
    h = int(round((sc.x_center),0))
    k = int(round((sc.y_center),0))

    # Assign nominal semi-major and semi-minor axes
    rx = int(round((sc.pixels[1].max() - h),0))
    ry = int(round((sc.pixels[0].max() - k),0))

    # Create a rectangular bounding box around both clusters
    xmin = h-rx
    xmax = xmin+2*rx+1
    ymin = k-ry
    ymax = ymin+2*ry+1
    xrange = range(xmin,xmax,1)
    yrange = range(ymin,ymax,1)
    if verbose: print('MERGE> ranges for pixel selection: ',xrange,yrange)
    (XX,YY) = np.meshgrid(xrange,yrange)

    # Calculate which bounding box pixels lie within the ellipse region:
    separation = ((XX - h)**2/(rx*rx)) + ((YY-k)**2/(ry*ry))
    idx = np.where(separation <= 1.0)
    sc.pixels = (YY[idx],XX[idx])
    if verbose: print('MERGE> sc reselected pixels: ',sc.pixels, len(sc.pixels[0]))

    if verbose: print('MERGE> sc final: ',sc.summary())

    return sc

def mergeClusters(clusters, priorityMap, priorityThreshold, verbose=False):
    """Function to create a supercluster from a set of
    smaller clusters by combining their pixel arrays.
    Overlapping pixels are identified to avoid inflating
    pixel arrays with duplicated pixels.
    The bounding box of the cluster is calculated, although
    the actual pixel map included can be irregular.
    """

    # Merge the pixel arrays of all component clusters
    # Note the ordering of the pixel index arrays in cluster.pixels
    # is (y,x) in keeping with Python array handling
    sc = PixelCluster()
    xpixels = np.array([],dtype='int')
    ypixels = np.array([],dtype='int')
    for i,member in clusters.items():
        xpixels = np.concatenate((xpixels, member.pixels[1]))
        ypixels = np.concatenate((ypixels, member.pixels[0]))
    sc.pixels = (np.array(ypixels),np.array(xpixels))
    if verbose: print('MERGE> sc concat pixels: ',sc.pixels)

    # Use the combined pixel arrays to calculate a rectangular bounding box
    sc.calcClusterRanges()
    sc.calcCentroid()
    if verbose: print('MERGE> sc init: ',sc.summary())

    # Select from the priorityMap all HEALpixels within the bounding box
    # with a priority greater than the selection threshold
    xrange = range(sc.xmin,sc.xmax+1,1)
    yrange = range(sc.ymin,sc.ymax+1,1)
    if verbose: print('MERGE> ranges for pixel selection: ',xrange,yrange)
    (XX,YY) = np.meshgrid(xrange,yrange)
    boxPixels = priorityMap[(YY,XX)]
    idx = np.where(boxPixels > priorityThreshold)
    sc.pixels = (YY[idx],XX[idx])
    if verbose: print('MERGE> sc reselected pixels: ',sc.pixels, len(sc.pixels[0]))

    # Since the pixel map has changed, the centroid and bounding box need
    # updating as well
    sc.calcClusterRanges()
    sc.calcCentroid()
    if verbose: print('MERGE> sc final: ',sc.summary())

    return sc

def reviewSuperCluster(sc, clusters, priorityMap, fracIncrease):
    """
    Function to compare the new supercluster with the component clusters
    it was built from and determine whether the merge should be accepted.
    A merge is accepted based on whether it results in a significant increase
    in the total HEALpixel priority included in the pixel map, summed over
    all pixels.
    """

    # Check there are actually valid pixels within the SC:
    if len(sc.pixels) == 0:
        print('Supercluster has no valid pixels')
        return False

    # Calculate the summed HEALpixel priority of the supercluster and
    # component clusters
    sc.sumPixelPriority(priorityMap)
    clusterSum = 0.0
    for i,member in clusters.items():
        member.sumPixelPriority(priorityMap)
        clusterSum += member.priority

    # Determine whether the supercluster is preferable to the
    # separate clusters
    sumThreshold = clusterSum + clusterSum*fracIncrease

    print('SC total priority='+str(sc.priority))
    print('Combined cluster priority='+str(clusterSum))
    print('Priority threshold = '+str(sumThreshold))
    if sc.priority >= sumThreshold:
        return True
    else:
        return False

def plotClustersMap(params, clusters, priorityMap, it):

    clusterMap = np.zeros(priorityMap.shape)
    clusterPositions = np.zeros((len(clusters),2))

    for i,member in clusters.items():
        clusterMap[member.pixels] = priorityMap[member.pixels]
        clusterPositions[i,0] = member.x_center
        clusterPositions[i,1] = member.y_center

    fig = plt.figure(7,(10,10))
    imgplot = plt.imshow(clusterMap)
    plt.plot(clusterPositions[:,0], clusterPositions[:,1], 'r.')
    plt.title('Galactic science priority map with clusters marked')
    plt.savefig(path.join(params['output_dir'],'map_merged_clusters_priority_'+str(it)+'.png'))
    plt.close(7)

def calcAreaOfClusters(clusters,pixArea):
    area = 0.0
    npixels = 0
    for i, member in clusters.items():
        area += member.calcPixelArea(pixArea)
        npixels += len(member.pixels[0])

    print('Total number of pixels selected: '+str(npixels))
    print('Total area of pixel clusters selected: '+str(round(area,1))+'sq deg')

def calcPriorityofClusters(clusters, priorityMap):
    sumPriority = 0.0
    for i, member in clusters.items():
        sumPriority += priorityMap[member.pixels].sum()
        member.calcPixelPriorityStats(priorityMap)
        print('Cluster '+str(i)+', '+str(len(member.pixels[0]))+\
                ' pixels avg priority '+str(member.avgPriority)+' with stddev '+\
                str(member.stdPriority)+' and max priority '+str(member.maxPriority))

    print('Total priority of the pixel clusters selected: '+\
            str(round(sumPriority,1)))

def filterClusters(clusters, priorityThreshold, priorityMap):

    # Filter out low-priority clusters
    newClusterList = []
    for i, member in clusters.items():
        member.calcPixelPriorityStats(priorityMap)
        if (member.avgPriority) >= priorityThreshold:
            newClusterList.append(member)

    # Re-index the revised dictionary of clusters
    newClusters = {}
    for i in range(0,len(newClusterList),1):
        newClusters[i] = newClusterList[i]

    print('Filteration eliminated '+str(len(clusters)-len(newClusters))+\
            ' clusters out of the original '+str(len(clusters)))

    return newClusters

def outputClusterMap(params, clusters, pixelMetaData,
                     priorityThreshold, avgPixelThreshold):

    selectedPixels = np.zeros(NPIX)
    plotPixels = np.zeros(NPIX)
    for i,member in clusters.items():
        for j in range(0,len(member.pixels[0]),1):
            y = member.pixels[0][j]
            x = member.pixels[1][j]
            # Note that where the boundaries of HEALpixels and Mollweide pixels
            # don't perfectly line up, multiple hits are possible.  In this case
            # all selected pixels are included, since this function assigns
            # rather than accumulating the priority values.
            xdx = np.where(pixelMetaData['image_pixel_x'] == x)[0]
            ydx = np.where(pixelMetaData['image_pixel_y'] == y)[0]
            idx = list(set(xdx).intersection(set(ydx)))
            hpids = pixelMetaData['HPid'][idx]
            selectedPixels[hpids] = pixelMetaData['priority'][idx]
            plotPixels[hpids] = 1.0

    hdr = fits.Header()
    hdr['NSIDE'] = NSIDE
    hdr['NPIX'] = NPIX
    hdr['MAPTITLE'] = 'galplane_survey_region'
    phdu = fits.PrimaryHDU(header=hdr)
    c1 = fits.Column(name='pixelPriority', array=selectedPixels, format='E')
    hdu = fits.BinTableHDU.from_columns([c1])
    hdul = fits.HDUList([phdu,hdu])

    file_path = path.join(params['output_dir'],
                'aggregated_priority_map_p'+str(round(priorityThreshold,2))
                +'_a'+str(round(avgPixelThreshold,2))+'_'+params['suffix']+'.fits')
    hdul.writeto(file_path, overwrite=True)
    print('Output the aggregated priority map datatable to '+file_path)

    # Output the same data as an Healpy mollweide plot for reference:
    fig = plt.figure(8,(10,10))
    plotPriority = False
    if plotPriority:
        hp.mollview(selectedPixels,title='Galactic Science Region of Interest',
                cmap=cm.plasma)
    else:
        hp.mollview(plotPixels,title='Galactic Science Region of Interest',
                cmap=cm.plasma)
    hp.graticule()
    plt.tight_layout()
    file_path = file_path.replace('.fits','.png')
    plt.savefig(file_path)
    plt.close(8)

    return selectedPixels

def outputClusterMapDataTables(params, selectedPixels,
                     priorityThreshold, avgPixelThreshold):
    """Function to loop over the priority map data tables in each filter, and
    output revised datatables containing the same priority pixel data for
    the combined map for the selectedPixels only"""

    for f in ['u','g','r','i','z','y']:
        data_file = path.join(params['input_dir'], params['data_table_root']+'_'+f+'.fits')
        with fits.open(data_file) as hdul:
            mapTable = hdul[1].data

        revisedMap = np.zeros(NPIX)
        idx = np.where(selectedPixels > 0.0)[0]
        revisedMap[idx] = mapTable.combined_map[idx]

        hdr = fits.Header()
        hdr['NSIDE'] = NSIDE
        hdr['NPIX'] = NPIX
        hdr['MAPTITLE'] = 'galplane_survey_region'
        phdu = fits.PrimaryHDU(header=hdr)
        c1 = fits.Column(name='combined_map', array=revisedMap, format='E')
        hdu = fits.BinTableHDU.from_columns([c1])
        hdul = fits.HDUList([phdu,hdu])
        file_path = path.join(params['output_dir'],
                    'aggregated_priority_map_data_p'+str(round(priorityThreshold,2))
                    +'_a'+str(round(avgPixelThreshold,2))+'_'+f+'.fits')
        hdul.writeto(file_path, overwrite=True)
        print('Output the aggregated data table datatable to '+file_path)

def get_args():
    params = {}

    if len(argv) == 1:
        params['input_dir'] = input('Please enter the path to the image to be analysed: ')
        params['image_path'] = input('Please enter the path of the image to be analysed: ')
        params['data_table_root'] = input('Please enter the rootname of the map data tables, minus suffix: ')
        params['suffix'] = input('Please enter the file suffix to use for output (e.g. filter): ')
        params['output_dir'] = input('Please enter the path to the output directory: ')
        params['priority_min'] = float(input('Please enter the minimum of the priority threshold range: '))
        params['priority_max'] = float(input('Please enter the maximum of the priority threshold range: '))
        params['priority_step'] = float(input('Please enter the step in the priority threshold range: '))
    else:
        params['input_dir'] = argv[1]
        params['image_path'] = argv[2]
        params['data_table_root'] = argv[3]
        params['suffix'] = argv[4]
        params['output_dir'] = argv[5]
        params['priority_min'] = float(argv[6])
        params['priority_max'] = float(argv[7])
        params['priority_step'] = float(argv[8])

    return params


if __name__ == '__main__':
    params = get_args()
    analyse_image(params)
