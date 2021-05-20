import numpy as np
import xarray as xr
import pandas as pd




# calculate the clim
def ts2clm(ts,percentile=90,windowHalfWidth=5,smoothPercentile=True,smoothPercentileWidth=31,maxgap=2):
    """ Calculate a climatology from a temperature time series.
        time series must be a continous vector of single temperature value for each day
        
        :param xr.DataArray ts: temp array with coordinates TIME in datetime[64]
        :param percentile: The percentile used for threshold calculation
        :param windowHalfWidth: The size of the window used for caculations either side of the current value
        :param smoothPercentile: Smooth the output using running mean
        :param smoothPercentileWidth: Number of days for smoothing
        :param maxgap: Maximum gap to fill
        :return: Xarray Dataset with seas (cilmatology) and thresh (threshold)
    """
    # put year and dayof year in the coordinates
    ts.coords['year']=ts.TIME.dt.year
    if not 'dayofyear' in  ts.coords:
        ts.coords['dayofyear']=ts.TIME.dt.dayofyear
    ts.coords['dayofyear']=xr.where((ts.dayofyear>59) & (~ts.TIME.dt.is_leap_year),ts.dayofyear+1,ts.dayofyear)
    t1=ts.rolling(TIME=1+windowHalfWidth*2,min_periods=1,center=True).construct("window_dim").set_index(TIME=['dayofyear','year']).unstack()
    seas =t1.reduce(np.nanmean,dim=('year','window_dim'))
    ts[ts.dayofyear==60] =np.nan
    seas = seas[seas.dayofyear!=60].interp(dayofyear=range(1,367)) 
    thresh = t1.reduce(np.nanpercentile,dim=('year','window_dim'), q=percentile)
    thresh = thresh[thresh.dayofyear!=60].interp(dayofyear=range(1,367)) 
    if smoothPercentile:
        seas = seas.pad(dayofyear=smoothPercentileWidth*2, mode='wrap').rolling(dayofyear=smoothPercentileWidth,center=True).mean()[smoothPercentileWidth*2:-smoothPercentileWidth*2]
        thresh = thresh.pad(dayofyear=smoothPercentileWidth*2, mode='wrap').rolling(dayofyear=smoothPercentileWidth,center=True).mean()[smoothPercentileWidth*2:-smoothPercentileWidth*2]
    ds = xr.Dataset({'seas':seas,'thresh':thresh})
    return ds

def rle(inarray,minlength=5):
        """ run length encoding. Partial credit to R rle function. 
            Multi datatype arrays catered for including non Numpy
            returns: tuple (runlengths, startpositions, values) """
        n = len(inarray)
        if n == 0: 
            return (None, None, None)
        else:
            y = np.array(inarray[1:] != inarray[:-1])     # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)   # must include last element posi
            z = np.diff(np.append(-1, i))       # run lengths
            p = np.cumsum(np.append(0, z))[:-1] # positions
            z =z[inarray[i]]
            p =p[inarray[i]]
            
            mask = (z>=minlength)
            print(minlength)
            z = z[mask]
            p =p[mask]
            return(z, p)

def detectevent(observations,threshClim,clim,minDuration = 5,joinAcrossGaps = True,maxGap = 2,coldSpells = False):
    def process_event(item):
        if item.StartTime is None:
            return item
        # Get SST series during MHW event, relative to both threshold and to seasonal climatology
        tt_start =(observations.TIME==np.datetime64(item.StartTime)).argmax().values
        tt_end=(observations.TIME==np.datetime64(item.EndTime)).argmax().values
        item.index_start=tt_start
        item.index_end=tt_end
        temp_mhw = observations[tt_start:tt_end+1].swap_dims({'TIME':'dayofyear'})
        thresh_mhw = threshClim.sel(dayofyear=temp_mhw.dayofyear)
        seas_mhw = clim.sel(dayofyear=temp_mhw.dayofyear)
        mhw_relSeas = temp_mhw - seas_mhw
        mhw_relThresh = temp_mhw - thresh_mhw
        mhw_relThreshNorm = (temp_mhw - thresh_mhw) / (thresh_mhw - seas_mhw)
        mhw_abs = temp_mhw
        # Find peak
        tt_peak = mhw_relSeas.argmax().values
        #item.time_peak=item.time_start'][ev] + tt_peak)
        item.time_peak=mhw_relSeas.idxmax()
        item.date_peak=mhw_relSeas.idxmax().values.astype('M8[D]')
        # MHW Duration
        item.duration=len(mhw_relSeas)
        # MHW Intensity metrics
        item.intensity_max=mhw_relSeas.max()
        item.intensity_mean=mhw_relSeas.mean()
        item.intensity_var=np.sqrt(mhw_relSeas.var())
        item.intensity_cumulative=mhw_relSeas.sum()
        item.intensity_max_relThresh=mhw_relThresh[tt_peak]
        item.intensity_mean_relThresh=mhw_relThresh.mean()
        item.intensity_var_relThresh=np.sqrt(mhw_relThresh.var())
        item.intensity_cumulative_relThresh=mhw_relThresh.sum()
        item.intensity_max_abs=mhw_abs[tt_peak]
        item.intensity_mean_abs=mhw_abs.mean()
        item.intensity_var_abs=np.sqrt(mhw_abs.var())
        item.intensity_cumulative_abs=mhw_abs.sum()
        # Fix categories
        tt_peakCat = mhw_relThreshNorm.argmax()
        cats = np.floor(1. + mhw_relThreshNorm)
        item.category=categories[np.min([cats[tt_peakCat], 4]).astype(int) - 1]
        item.duration_moderate=np.sum(cats == 1.)
        item.duration_strong=np.sum(cats == 2.)
        item.duration_severe=np.sum(cats == 3.)
        item.duration_extreme=np.sum(cats >= 4.)
        
        # Rates of onset and decline
        # Requires getting MHW strength at "start" and "end" of event (continuous: assume start/end half-day before/after first/last point)
        if tt_start > 0:
            mhw_relSeas_start = 0.5*(mhw_relSeas[0] + observations[tt_start-1] - clim.sel(dayofyear=observations[tt_start-1].dayofyear))
            item.rate_onset=(mhw_relSeas[tt_peak] - mhw_relSeas_start) / (tt_peak+0.5)
        else: # MHW starts at beginning of time series
            if tt_peak == 0: # Peak is also at begining of time series, assume onset time = 1 day
                item.rate_onset=(mhw_relSeas[tt_peak] - mhw_relSeas[0]) / 1.
            else:
                item.rate_onset=(mhw_relSeas[tt_peak] - mhw_relSeas[0]) / tt_peak
        if tt_end < len(observations)-1:
            mhw_relSeas_end = 0.5*(mhw_relSeas[-1] + observations[tt_end+1] -clim.sel(dayofyear=observations[tt_end+1].dayofyear))
            item.rate_decline=(mhw_relSeas[tt_peak] - mhw_relSeas_end) / (tt_end-tt_start-tt_peak+0.5)
        else: # MHW finishes at end of time series
            if tt_peak == len(observations)-1: # Peak is also at end of time series, assume decline time = 1 day
                item.rate_decline=(mhw_relSeas[tt_peak] - mhw_relSeas[-1]) / 1.
            else:
                item.rate_decline=(mhw_relSeas[tt_peak] - mhw_relSeas[-1]) / (tt_end-tt_start-tt_peak)
        return item
        
    # convert observations to a
    categories = np.array(['Moderate', 'Strong', 'Severe', 'Extreme'])
    observations.coords['dayofyear']=observations.TIME.dt.dayofyear
    observations.coords['dayofyear']=xr.where((observations.dayofyear>59) & (~observations.TIME.dt.is_leap_year),observations.dayofyear+1,observations.dayofyear)
    exceed_bool = (observations.groupby('dayofyear')-threshClim)>0
    runs, index =rle(exceed_bool.values,minDuration)
    events =pd.DataFrame.from_records({'StartTime':exceed_bool.isel(TIME=index).TIME.values,'EndTime':exceed_bool.isel(TIME=index+runs-1).TIME.values},columns=['StartTime','EndTime'])
    events =events.groupby((events.StartTime - events.EndTime.shift() >pd.to_timedelta(maxGap,'D')).cumsum()).agg({'StartTime':'min', 'EndTime':'max'})[['StartTime','EndTime']]
    events['time_peak'] = 0
    events['date_peak'] = 0 # datetime format
    events['index_start'] = 0
    events['index_end'] = 0
    events['index_peak'] = 0
    events['duration'] = 0 # [days]
    events['duration_moderate'] = 0 # [days]
    events['duration_strong'] = 0 # [days]
    events['duration_severe'] = 0 # [days]
    events['duration_extreme'] = 0 # [days]
    events['intensity_max'] = 0 # [deg C]
    events['intensity_mean'] = 0 # [deg C]
    events['intensity_var'] = 0 # [deg C]
    events['intensity_cumulative'] = 0 # [deg C]
    events['intensity_max_relThresh'] = 0 # [deg C]
    events['intensity_mean_relThresh'] = 0 # [deg C]
    events['intensity_var_relThresh'] = 0 # [deg C]
    events['intensity_cumulative_relThresh'] = 0 # [deg C]
    events['intensity_max_abs'] = 0 # [deg C]
    events['intensity_mean_abs'] = 0 # [deg C]
    events['intensity_var_abs'] = 0 # [deg C]
    events['intensity_cumulative_abs'] = 0 # [deg C]
    events['category'] = 0
    events['rate_onset'] = 0 # [deg C / day]
    events['rate_decline'] = 0 # [deg C / day]
    events=events.apply(process_event,axis=1)
    return events    


def synthclim(startdate="1984-01-01",enddate="2014-12-31"):
    """ generate a synthetic climatology.
        
        :param str startdate: start date of series eg. 1984-01-01
        :param str enddate: end date of series eg. 2014-01-01
    """
    dates =pd.date_range(start=startdate, end=enddate)
    sst = 0.5*np.cos(dates.dayofyear.values*2*np.pi/365.25)
    a = 0.85 # autoregressive parameter
    for i in range(1,len(sst)):
        sst[i] = a*sst[i-1] + 0.75*np.random.randn() +sst[i]
    sst = sst - sst.min() + 5.
    # Generate time vector using datetime format (January 1 of year 1 is day 1)
    da =xr.DataArray(sst , dims=['TIME'],coords={"TIME": ("TIME", dates, {'units': 'days since '+startdate})},name='TEMP',attrs={'units':'Deg C'})
    #da.coords['time']['attr']={'units':'days since '+startdate}
    return da 


