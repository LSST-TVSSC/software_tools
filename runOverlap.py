#!/usr/bin/env python
import sys
sys.path.append("/home/mike/lsst/metrics")
#print sys.path
import numpy as np 
import matplotlib.pyplot as plt
#%matplotlib inline
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
from lsst.sims.maf.plots import PlotHandler
from SpacialOverlapMetric import SpacialOverlapMetric
database='astro-lsst-01_2022.db'
databasename='astro-lsst-01_2022'
opsdb = db.OpsimDatabase(database)
OutDir='Overlap_OGLE'
region_name='stripe82'
region_name='OGLE_disk'
spacialoverlapmetric=SpacialOverlapMetric(region_name=region_name)
slicer = slicers.OpsimFieldSlicer()
slicer = slicers.HealpixSlicer(nside=64)
Title='Spacial Overlap Metric for '+databasename+'\n'+region_name

plotDict={'title':Title}
sqlconstraint = ''
SOmetric = metricBundles.MetricBundle(spacialoverlapmetric, slicer, sqlconstraint, plotDict=plotDict, runName=databasename)
summaryMetrics = [metrics.SumMetric()]
SOmetric.setSummaryMetrics(summaryMetrics)
bundleDict = {'spacial':SOmetric}
resultsDb=db.ResultsDb(outDir=OutDir)
group = metricBundles.MetricBundleGroup(bundleDict, opsdb, outDir=OutDir, resultsDb=resultsDb)
group.runAll()
group.plotAll(closefigs=False)
print("Summary", SOmetric.summaryValues)
