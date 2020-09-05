import numpy as np
import xarray as xr
import pandas as pd


# calculate the clim
def ts2clm(ts,percentile=90,windowHalfWidth=5,smoothPercentile=True,smoothPercentileWidth=31,maxgap=2,paddays=31):
    # put year and dayof year in the coordinates
    ts.coords['year']=ts.TIME.dt.year
    ts.coords['dayofyear']=ts.TIME.dt.dayofyear
    #lets make Feb 29th on the non leap years 
    #unstack the data into table year vs dayofyear
    ts =ts.set_index(TIME=['dayofyear','year']).unstack()
    #pad ts with an extra month at each end and construct the dataset with halfwindow
    t1 =ts.pad(dayofyear=paddays, mode='wrap').rolling(dayofyear=1+windowHalfWidth*2,min_periods=1,center=True).construct("window_dim")
    
    #take the mean of each bin
    seas =t1.reduce(np.nanmean,dim=('year','window_dim'))
    #interplate the Feb 29th
    seas = xr.where(seas.dayofyear==60,np.nan,seas)[paddays:-paddays].interpolate_na(dim='dayofyear',max_gap=maxgap)
    thresh = t1.reduce(np.nanpercentile,dim=('year','window_dim'), q=percentile) #
    thresh = xr.where(thresh.dayofyear==60,np.nan,thresh)[paddays:-paddays].interpolate_na(dim='dayofyear',max_gap=maxgap)
    
    if smoothPercentile:
        seas = seas.pad(dayofyear=paddays, mode='wrap').rolling(dayofyear=smoothPercentileWidth,center=True).mean()[paddays:-paddays]
        thresh = thresh.pad(dayofyear=paddays, mode='wrap').rolling(dayofyear=smoothPercentileWidth,center=True).mean()[paddays:-paddays]
    ds = xr.Dataset({'seas':seas,'thresh':thresh})
    return ds

def synthclim(startdate="1984-01-01",enddate="2014-12-31"):
    dates =pd.date_range(start=startdate, end=enddate)
    sst = 0.5*np.cos(dates.dayofyear.values*2*np.pi/365.25)
    a = 0.85 # autoregressive parameter
    for i in range(1,len(sst)):
        sst[i] = a*sst[i-1] + 0.75*np.random.randn() +sst[i]
    sst = sst - sst.min() + 5.
    # Generate time vector using datetime format (January 1 of year 1 is day 1)
    da =xr.DataArray(sst , dims={"time": dates},attrs={'units':'days since '+startdate},name='sst')
    return da 
    