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
from EventFractionMetric import EventFractionMetric
database='astro-lsst-01_2022.db'
databasename='astro-lsst-01_2022'
opsdb = db.OpsimDatabase(database)
OutDir='EventFraction'
kwargs={'frac':0.001, 'event_count':3}
eventfractionmetric=EventFractionMetric(**kwargs)
slicer = slicers.OpsimFieldSlicer()
Title='Event Fraction Metric for '+databasename

plotDict={'title':Title, 'logScale':True}
sqlconstraint = ''
eventfractions = metricBundles.MetricBundle(eventfractionmetric, slicer, sqlconstraint, plotDict=plotDict, runName=databasename)
bundleDict = {'eventfraction':eventfractions}
resultsDb=db.ResultsDb(outDir=OutDir)
group = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=OutDir, resultsDb=resultsDb)
group.runAll()
group.plotAll(closefigs=False)
