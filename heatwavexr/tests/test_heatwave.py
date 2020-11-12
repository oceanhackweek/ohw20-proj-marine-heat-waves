import heatwavexr as hw

psst = hw.synthclim()
test = hw.ts2clm(psst)
sst,mhw = hw.detect(psst)
result = hw.detectevent(psst,test.thresh,test.seas)
print(result)
