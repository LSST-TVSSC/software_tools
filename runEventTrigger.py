#!/usr/bin/env python
import sys
sys.path.append("/home/mike/lsst/metrics") # modify path as needed
import numpy as np 
import matplotlib.pyplot as plt
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
from lsst.sims.maf.plots import PlotHandler
from EventTriggerMetric import EventTriggerMetric
database='astro-lsst-01_2022.db'
databasename='astro-lsst-01_2022'
opsdb = db.OpsimDatabase(database)
OutDir='Microlens/Exoplanet'
kwargs={'DelMin':48, 'DelMax':168} # Sample stellar microlensing parameter
kwargs={'DelMin':1, 'DelMax':48} # Sample exoplanet microlensing parameter
eventtriggermetric=EventTriggerMetric(**kwargs)
slicer = slicers.OpsimFieldSlicer()
Title='EventTrigger Metric for '+databasename

#plotDict={'colorMin':0.0, 'colorMax':1.0, 'xlabel':'Periodogram Purity Function', 'title':Title}
plotDict={'title':Title, 'logScale':True, 'colorMin':0.00001, 'xlabel':'Fraction'}
sqlconstraint = ''
eventtrigger = metricBundles.MetricBundle(eventtriggermetric, slicer, sqlconstraint, plotDict=plotDict, runName=databasename)
bundleDict = {'EventTrigger':eventtrigger}
resultsDb=db.ResultsDb(outDir=OutDir)
group = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=OutDir, resultsDb=resultsDb)
group.runAll()
group.plotAll(closefigs=False)
