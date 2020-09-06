import heatwavexr as hw

psst = hw.synthclim(enddate='1986-12-31')
test = hw.ts2clm(psst)
sst = hw.detect(psst)