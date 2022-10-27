from os import path
from sys import argv
import numpy as np
import objdetect_galplane_map
import matplotlib.pyplot as plt

def test_calcImagePixels():

    FoV = objdetect_galplane_map.RubinFOV(nPixels=13)
    image = np.ones((100,100))
    xc = 25.0
    yc = 50.0

    pixels = FoV.calcImagePixels(image,xc,yc)

    assert len(pixels) == 2
    for arr in pixels:
        assert type(arr) == type(np.zeros(1))
        assert len(arr) == 13
    np.testing.assert_almost_equal(np.median(pixels[0]),xc,1.0)
    np.testing.assert_almost_equal(np.median(pixels[1]),yc,1.0)

def test_merge_clusters():
    FoV = objdetect_galplane_map.RubinFOV(nPixels=13)
    priorityMap = np.ones((100,200))

    cluster1 = objdetect_galplane_map.PixelCluster(45.0,45.0)
    cluster1.pixels = FoV.calcImagePixels(priorityMap,
                                            cluster1.x_center,
                                            cluster1.y_center)
    cluster1.calcClusterRanges()
    print('Cluster 1: ',cluster1.summary())

    cluster2 = objdetect_galplane_map.PixelCluster(60.0,55.0)
    cluster2.pixels = FoV.calcImagePixels(priorityMap,
                                            cluster2.x_center,
                                            cluster2.y_center)
    cluster2.calcClusterRanges()
    print('Cluster 2: ',cluster2.summary())

    neighbors = {0: cluster1, 1: cluster2}

    sc = objdetect_galplane_map.mergeClusters(neighbors,
                                              priorityMap,
                                              0.35,
                                              verbose=True)
    print('Supercluster: ',sc.summary())

    fig = plt.figure(1,(10,10))
    mapImage = np.zeros(priorityMap.shape)
    for i,cluster in neighbors.items():
        mapImage[cluster.pixels] += priorityMap[cluster.pixels]
    mapImage[sc.pixels] += priorityMap[sc.pixels]

    imgplot = plt.imshow(mapImage)
    for i,cluster in neighbors.items():
        plt.plot(cluster.x_center, cluster.y_center, 'r.')
    plt.plot(sc.x_center, sc.y_center, 'b+')
    plt.savefig('test_map.png')
    plt.close(1)

    expected_xmin = min(cluster1.pixels[1].min(), cluster2.pixels[1].min())
    expected_xmax = max(cluster1.pixels[1].max(), cluster2.pixels[1].max())
    expected_ymin = min(cluster1.pixels[0].min(), cluster2.pixels[0].min())
    expected_ymax = max(cluster1.pixels[0].max(), cluster2.pixels[0].max())
    expected_xrange = range(expected_xmin,expected_xmax+1,1)
    expected_yrange = range(expected_ymin,expected_ymax+1,1)
    expected_npixels = len(expected_xrange)*len(expected_yrange)

    assert sc.x_center == (cluster1.x_center+cluster2.x_center)/2.0
    assert sc.y_center == (cluster1.y_center+cluster2.y_center)/2.0
    assert sc.xmin == expected_xmin
    assert sc.xmax == expected_xmax
    assert sc.ymin == expected_ymin
    assert sc.ymax == expected_ymax
    assert len(sc.pixels[1]) == expected_npixels

def test_rubin_fov():

    pixScale = 1.0
    fov = objdetect_galplane_map.RubinFOV(pixScale)
    expected_footprint = np.array(  [[0,0,1,0,0],
                                     [0,1,1,1,0],
                                     [1,1,1,1,1],
                                     [0,1,1,1,0],
                                     [0,0,1,0,0]])
    assert (fov.footprint == expected_footprint).all()

    pixScale = 0.5
    fov = objdetect_galplane_map.RubinFOV(pixScale)
    expected_footprint = np.array(  [[0,0,0,1,0,0,0],
                                     [0,0,1,1,1,0,0],
                                     [0,1,1,1,1,1,0],
                                     [1,1,1,1,1,1,1],
                                     [0,1,1,1,1,1,0],
                                     [0,0,1,1,1,0,0],
                                     [0,0,0,1,0,0,0]] )
    assert (fov.footprint == expected_footprint).all()

if __name__ == '__main__':
    #test_calcImagePixels()
    #test_merge_clusters()
    test_rubin_fov()
