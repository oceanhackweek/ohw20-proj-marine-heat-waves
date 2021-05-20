import heatwavexr as hw

psst = hw.synthclim()
sst,mhw = hw.mhwdetect(psst)
test = hw.ts2clm(psst)

result = hw.detectevent(psst,test.thresh,test.seas)
print(result)
