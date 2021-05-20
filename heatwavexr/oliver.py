import numpy as np
import xarray as xr
from datetime import date
import scipy.ndimage as ndimage

def mhwdetect(t, pctile=90, windowHalfWidth=5, smoothPercentile=True, smoothPercentileWidth=31, minDuration=5, joinAcrossGaps=True, maxGap=2, maxPadLength=False, coldSpells=False, alternateClimatology=False, Ly=False):
    '''

    Applies the Hobday et al. (2016) marine heat wave definition to an input time
    series of temp ('temp') along with a time vector ('t'). Outputs properties of
    all detected marine heat waves.

    Inputs:

      t       Time vector, in datetime format (e.g., date(1982,1,1).toordinal())
              [1D numpy array of length T]
      temp    Temperature vector [1D numpy array of length T]

    Outputs:

      mhw     Detected marine heat waves (MHWs). Each key (following list) is a
              list of length N where N is the number of detected MHWs:
 
        'time_start'           Start time of MHW [datetime format]
        'time_end'             End time of MHW [datetime format]
        'time_peak'            Time of MHW peak [datetime format]
        'date_start'           Start date of MHW [datetime format]
        'date_end'             End date of MHW [datetime format]
        'date_peak'            Date of MHW peak [datetime format]
        'index_start'          Start index of MHW
        'index_end'            End index of MHW
        'index_peak'           Index of MHW peak
        'duration'             Duration of MHW [days]
        'intensity_max'        Maximum (peak) intensity [deg. C]
        'intensity_mean'       Mean intensity [deg. C]
        'intensity_var'        Intensity variability [deg. C]
        'intensity_cumulative' Cumulative intensity [deg. C x days]
        'rate_onset'           Onset rate of MHW [deg. C / days]
        'rate_decline'         Decline rate of MHW [deg. C / days]

        'intensity_max_relThresh', 'intensity_mean_relThresh', 'intensity_var_relThresh', 
        and 'intensity_cumulative_relThresh' are as above except relative to the
        threshold (e.g., 90th percentile) rather than the seasonal climatology

        'intensity_max_abs', 'intensity_mean_abs', 'intensity_var_abs', and
        'intensity_cumulative_abs' are as above except as absolute magnitudes
        rather than relative to the seasonal climatology or threshold

        'category' is an integer category system (1, 2, 3, 4) based on the maximum intensity
        in multiples of threshold exceedances, i.e., a value of 1 indicates the MHW
        intensity (relative to the climatology) was >=1 times the value of the threshold (but
        less than 2 times; relative to climatology, i.e., threshold - climatology).
        Category types are defined as 1=strong, 2=moderate, 3=severe, 4=extreme. More details in
        Hobday et al. (in prep., Oceanography). Also supplied are the duration of each of these
        categories for each event.

        'n_events'             A scalar integer (not a list) indicating the total
                               number of detected MHW events

      clim    Climatology of SST. Each key (following list) is a seasonally-varying
              time series [1D numpy array of length T] of a particular measure:

        'thresh'               Seasonally varying threshold (e.g., 90th percentile)
        'seas'                 Climatological seasonal cycle
        'missing'              A vector of TRUE/FALSE indicating which elements in 
                               temp were missing values for the MHWs detection

    Options:

      climatologyPeriod      Period over which climatology is calculated, specified
                             as list of start and end years. Default is to calculate
                             over the full range of years in the supplied time series.
                             Alternate periods suppled as a list e.g. [1983,2012].
      pctile                 Threshold percentile (%) for detection of extreme values
                             (DEFAULT = 90)
      windowHalfWidth        Width of window (one sided) about day-of-year used for
                             the pooling of values and calculation of threshold percentile
                             (DEFAULT = 5 [days])
      smoothPercentile       Boolean switch indicating whether to smooth the threshold
                             percentile timeseries with a moving average (DEFAULT = True)
      smoothPercentileWidth  Width of moving average window for smoothing threshold
                             (DEFAULT = 31 [days])
      minDuration            Minimum duration for acceptance detected MHWs
                             (DEFAULT = 5 [days])
      joinAcrossGaps         Boolean switch indicating whether to join MHWs
                             which occur before/after a short gap (DEFAULT = True)
      maxGap                 Maximum length of gap allowed for the joining of MHWs
                             (DEFAULT = 2 [days])
      maxPadLength           Specifies the maximum length [days] over which to interpolate
                             (pad) missing data (specified as nans) in input temp time series.
                             i.e., any consecutive blocks of NaNs with length greater
                             than maxPadLength will be left as NaN. Set as an integer.
                             (DEFAULT = False, interpolates over all missing values).
      coldSpells             Specifies if the code should detect cold events instead of
                             heat events. (DEFAULT = False)
      alternateClimatology   Specifies an alternate temperature time series to use for the
                             calculation of the climatology. Format is as a list of numpy
                             arrays: (1) the first element of the list is a time vector,
                             in datetime format (e.g., date(1982,1,1).toordinal())
                             [1D numpy array of length TClim] and (2) the second element of
                             the list is a temperature vector [1D numpy array of length TClim].
                             (DEFAULT = False)
      Ly                     Specifies if the length of the year is < 365/366 days (e.g. a 
                             360 day year from a climate model). This affects the calculation
                             of the climatology. (DEFAULT = False)

    Notes:

      1. This function assumes that the input time series consist of continuous daily values
         with few missing values. Time ranges which start and end part-way through the calendar
         year are supported.

      2. This function supports leap years. This is done by ignoring Feb 29s for the initial
         calculation of the climatology and threshold. The value of these for Feb 29 is then
         linearly interpolated from the values for Feb 28 and Mar 1.

      3. The calculation of onset and decline rates assumes that the heat wave started a half-day
         before the start day and ended a half-day after the end-day. (This is consistent with the
         duration definition as implemented, which assumes duration = end day - start day + 1.)

      4. For the purposes of MHW detection, any missing temp values not interpolated over (through
         optional maxPadLLength) will be set equal to the seasonal climatology. This means they will
         trigger the end/start of any adjacent temp values which satisfy the MHW criteria.

      5. If the code is used to detect cold events (coldSpells = True), then it works just as for heat
         waves except that events are detected as deviations below the (100 - pctile)th percentile
         (e.g., the 10th instead of 90th) for at least 5 days. Intensities are reported as negative
         values and represent the temperature anomaly below climatology.

    Written by Eric Oliver, Institue for Marine and Antarctic Studies, University of Tasmania, Feb 2015

    '''
    T = len(t)
    year = t.TIME.dt.year
    month = t.TIME.dt.month
    day = t.TIME.dt.day
    doy =t.TIME.dt.dayofyear
    if not 'dayofyear' in  t.coords:
        t.coords['dayofyear']=t.TIME.dt.dayofyear

    # Leap-year baseline for defining day-of-year values
    year_leapYear = 2012 # This year was a leap-year and therefore doy in range of 1 to 366
    t_leapYear = np.arange(date(year_leapYear, 1, 1).toordinal(),date(year_leapYear, 12, 31).toordinal()+1)
    dates_leapYear = [date.fromordinal(tt.astype(int)) for tt in t_leapYear]
    month_leapYear = np.zeros((len(t_leapYear)))
    day_leapYear = np.zeros((len(t_leapYear)))
    doy_leapYear = np.zeros((len(t_leapYear)))
    for tt in range(len(t_leapYear)):
        month_leapYear[tt] = date.fromordinal(t_leapYear[tt]).month
        day_leapYear[tt] = date.fromordinal(t_leapYear[tt]).day
        doy_leapYear[tt] = t_leapYear[tt] - date(date.fromordinal(t_leapYear[tt]).year,1,1).toordinal() + 1
    # Calculate day-of-year values
    for tt in range(T):
        doy.values[tt] = doy_leapYear[(month_leapYear == month[tt].values) * (day_leapYear == day[tt].values)]

    # Constants (doy values for Feb-28 and Feb-29) for handling leap-years
    feb28 = 59
    feb29 = 60


    # Length of climatological year
    lenClimYear = 366
    # Start and end indices
    clim_start = np.where(year == year[0])[0][0]
    clim_end = np.where(year == year[-1])[0][-1]
    # Inialize arrays
    thresh_climYear = np.NaN*np.zeros(lenClimYear)
    seas_climYear = np.NaN*np.zeros(lenClimYear)
    clim = {}
    TClim = len(t)
    clim['thresh'] = np.NaN*np.zeros(TClim)
    clim['seas'] = np.NaN*np.zeros(TClim)
    # Loop over all day-of-year values, and calculate threshold and seasonal climatology across years
    for d in range(1,lenClimYear+1):
        # Special case for Feb 29
        if d == feb29:
            continue
        # find all indices for each day of the year +/- windowHalfWidth and from them calculate the threshold
        tt0 = np.where(doy[clim_start:clim_end+1] == d)[0] 
        # If this doy value does not exist (i.e. in 360-day calendars) then skip it
        if len(tt0) == 0:
            continue
        tt = np.array([])
        for w in range(-windowHalfWidth, windowHalfWidth+1):
            tt = np.append(tt, clim_start+tt0 + w)
        tt = tt[tt>=0] # Reject indices "before" the first element
        tt = tt[tt<TClim] # Reject indices "after" the last element
        thresh_climYear[d-1] = np.nanpercentile(t[tt.astype(int)], pctile)
        seas_climYear[d-1] = np.nanmean(t[tt.astype(int)])
    # Special case for Feb 29
    thresh_climYear[feb29-1] = 0.5*thresh_climYear[feb29-2] + 0.5*thresh_climYear[feb29]
    seas_climYear[feb29-1] = 0.5*seas_climYear[feb29-2] + 0.5*seas_climYear[feb29]

    # Smooth if desired
    if smoothPercentile:
        # If the length of year is < 365/366 (e.g. a 360 day year from a Climate Model)
        if Ly:
            valid = ~np.isnan(thresh_climYear)
            thresh_climYear[valid] = runavg(thresh_climYear[valid], smoothPercentileWidth)
            valid = ~np.isnan(seas_climYear)
            seas_climYear[valid] = runavg(seas_climYear[valid], smoothPercentileWidth)
        # >= 365-day year
        else:
            thresh_climYear = runavg(thresh_climYear, smoothPercentileWidth)
            seas_climYear = runavg(seas_climYear, smoothPercentileWidth)

    # Generate threshold for full time series
    result =xr.Dataset()
    # procdue a sreies
    result['thresh'] = xr.DataArray(thresh_climYear[0:366],name='thresh',dims=['dayofyear'],coords={'dayofyear':range(1,367)})
    c =xr.DataArray(np.zeros(len(t)),coords = t.coords,dims=t.dims)
    result['thresh_full'] =c.groupby('dayofyear')+result['thresh']
    result['seas'] = xr.DataArray(seas_climYear[0:366],name='thresh',dims=['dayofyear'])
    result['seas_full'] =c.groupby('dayofyear')+result['seas']
    


    # Time series of "True" when threshold is exceeded, "False" otherwise
    
    exceed_bool = (t.groupby('dayofyear')-result['thresh'])>0
    # exceed_bool[tempt<=0] = False
    # exceed_bool[tempt>0] = True
    # Fix issue where missing temp vaues (nan) are counted as True
    #exceed_bool[np.isnan(exceed_bool)] = False
    # Find contiguous regions of exceed_bool = True
    events, n_events = ndimage.label(exceed_bool)
    events = xr.DataArray(events,dims=['TIME'],coords={"TIME":exceed_bool.TIME})
    mhw ={}
    mhw['time_start'] = [] # datetime format
    mhw['time_end'] = [] # datetime format
    mhw['time_peak'] = [] # datetime format
    mhw['date_start'] = [] # datetime format
    mhw['date_end'] = [] # datetime format
    mhw['date_peak'] = [] # datetime format
    mhw['index_start'] = []
    mhw['index_end'] = []
    mhw['index_peak'] = []
    mhw['duration'] = [] # [days]
    mhw['duration_moderate'] = [] # [days]
    mhw['duration_strong'] = [] # [days]
    mhw['duration_severe'] = [] # [days]
    mhw['duration_extreme'] = [] # [days]
    mhw['intensity_max'] = [] # [deg C]
    mhw['intensity_mean'] = [] # [deg C]
    mhw['intensity_var'] = [] # [deg C]
    mhw['intensity_cumulative'] = [] # [deg C]
    mhw['intensity_max_relThresh'] = [] # [deg C]
    mhw['intensity_mean_relThresh'] = [] # [deg C]
    mhw['intensity_var_relThresh'] = [] # [deg C]
    mhw['intensity_cumulative_relThresh'] = [] # [deg C]
    mhw['intensity_max_abs'] = [] # [deg C]
    mhw['intensity_mean_abs'] = [] # [deg C]
    mhw['intensity_var_abs'] = [] # [deg C]
    mhw['intensity_cumulative_abs'] = [] # [deg C]
    mhw['category'] = []
    mhw['rate_onset'] = [] # [deg C / day]
    mhw['rate_decline'] = [] # [deg C / day]
    # Find all MHW events of duration >= minDuration
    for ev in range(1,n_events+1):
        event_duration = (events == ev).sum()
        if event_duration < minDuration:
            continue
        mhw['time_start'].append(t[np.where(events == ev)[0][0]].TIME.values)
        mhw['time_end'].append(t[np.where(events == ev)[0][-1]].TIME.values)

    # Link heat waves that occur before and after a short gap (gap must be no longer than maxGap)
    if joinAcrossGaps:
        # Calculate gap length for each consecutive pair of events
        gaps = ((np.array(mhw['time_start'][1:]) - np.array(mhw['time_end'][0:-1])) / np.timedelta64(1, 'D'))-1
        if len(gaps) > 0:
            while gaps.min() <= maxGap:
                # Find first short gap
                ev = np.where(gaps <= maxGap)[0][0]
                # Extend first MHW to encompass second MHW (including gap)
                mhw['time_end'][ev] = mhw['time_end'][ev+1]
                # Remove second event from record
                del mhw['time_start'][ev+1]
                del mhw['time_end'][ev+1]
                # Calculate gap length for each consecutive pair of events
                gaps =((np.array(mhw['time_start'][1:]) - np.array(mhw['time_end'][0:-1])) / np.timedelta64(1, 'D'))-1
                if len(gaps) == 0:
                    break
    #
    # Initialize MHW output variable
    #


    # Calculate marine heat wave properties
    mhw['n_events'] = len(mhw['time_start'])
    categories = np.array(['Moderate', 'Strong', 'Severe', 'Extreme'])
    for ev in range(mhw['n_events']):
        mhw['date_start'].append(mhw['time_start'][ev].astype('M8[D]').astype('O'))
        mhw['date_end'].append(mhw['time_end'][ev].astype('M8[D]').astype('O'))
        # Get SST series during MHW event, relative to both threshold and to seasonal climatology
        #tt_start = np.where(t==mhw['time_start'][ev])[0][0]
        tt_start =(t.TIME==mhw['time_start'][ev]).argmax().values
        #tt_end = np.where(t==mhw['time_end'][ev])[0][0]
        tt_end=(t.TIME==mhw['time_end'][ev]).argmax().values
        mhw['index_start'].append(tt_start)
        mhw['index_end'].append(tt_end)
        temp_mhw = t[tt_start:tt_end+1]
        thresh_mhw = result['thresh_full'][tt_start:tt_end+1]
        seas_mhw = result['seas_full'][tt_start:tt_end+1]
        mhw_relSeas = temp_mhw - seas_mhw
        mhw_relThresh = temp_mhw - thresh_mhw
        mhw_relThreshNorm = (temp_mhw - thresh_mhw) / (thresh_mhw - seas_mhw)
        mhw_abs = temp_mhw
        # Find peak
        tt_peak = mhw_relSeas.argmax().values
        #mhw['time_peak'].append(mhw['time_start'][ev] + tt_peak)
        mhw['time_peak'].append(mhw_relSeas.idxmax())
        mhw['date_peak'].append(mhw_relSeas.idxmax().values.astype('M8[D]'))
        #mhw['index_peak'].append(tt_start + tt_peak)
        # MHW Duration
        mhw['duration'].append(len(mhw_relSeas))
        # MHW Intensity metrics
        mhw['intensity_max'].append(mhw_relSeas.max())
        mhw['intensity_mean'].append(mhw_relSeas.mean())
        mhw['intensity_var'].append(np.sqrt(mhw_relSeas.var()))
        mhw['intensity_cumulative'].append(mhw_relSeas.sum())
        mhw['intensity_max_relThresh'].append(mhw_relThresh[tt_peak])
        mhw['intensity_mean_relThresh'].append(mhw_relThresh.mean())
        mhw['intensity_var_relThresh'].append(np.sqrt(mhw_relThresh.var()))
        mhw['intensity_cumulative_relThresh'].append(mhw_relThresh.sum())
        mhw['intensity_max_abs'].append(mhw_abs[tt_peak])
        mhw['intensity_mean_abs'].append(mhw_abs.mean())
        mhw['intensity_var_abs'].append(np.sqrt(mhw_abs.var()))
        mhw['intensity_cumulative_abs'].append(mhw_abs.sum())
        # Fix categories
        tt_peakCat = mhw_relThreshNorm.argmax()
        cats = np.floor(1. + mhw_relThreshNorm)
        mhw['category'].append(categories[np.min([cats[tt_peakCat], 4]).astype(int) - 1])
        mhw['duration_moderate'].append(np.sum(cats == 1.))
        mhw['duration_strong'].append(np.sum(cats == 2.))
        mhw['duration_severe'].append(np.sum(cats == 3.))
        mhw['duration_extreme'].append(np.sum(cats >= 4.))
        
        # Rates of onset and decline
        # Requires getting MHW strength at "start" and "end" of event (continuous: assume start/end half-day before/after first/last point)
        if tt_start > 0:
            mhw_relSeas_start = 0.5*(mhw_relSeas[0] + t[tt_start-1] - clim['seas'][tt_start-1])
            mhw['rate_onset'].append((mhw_relSeas[tt_peak] - mhw_relSeas_start) / (tt_peak+0.5))
        else: # MHW starts at beginning of time series
            if tt_peak == 0: # Peak is also at begining of time series, assume onset time = 1 day
                mhw['rate_onset'].append((mhw_relSeas[tt_peak] - mhw_relSeas[0]) / 1.)
            else:
                mhw['rate_onset'].append((mhw_relSeas[tt_peak] - mhw_relSeas[0]) / tt_peak)
        if tt_end < T-1:
            mhw_relSeas_end = 0.5*(mhw_relSeas[-1] + t[tt_end+1] - clim['seas'][tt_end+1])
            mhw['rate_decline'].append((mhw_relSeas[tt_peak] - mhw_relSeas_end) / (tt_end-tt_start-tt_peak+0.5))
        else: # MHW finishes at end of time series
            if tt_peak == T-1: # Peak is also at end of time series, assume decline time = 1 day
                mhw['rate_decline'].append((mhw_relSeas[tt_peak] - mhw_relSeas[-1]) / 1.)
            else:
                mhw['rate_decline'].append((mhw_relSeas[tt_peak] - mhw_relSeas[-1]) / (tt_end-tt_start-tt_peak))

    # Flip climatology and intensties in case of cold spell detection
    if coldSpells:
        clim['seas'] = -1.*clim['seas']
        clim['thresh'] = -1.*clim['thresh']
        for ev in range(len(mhw['intensity_max'])):
            mhw['intensity_max'][ev] = -1.*mhw['intensity_max'][ev]
            mhw['intensity_mean'][ev] = -1.*mhw['intensity_mean'][ev]
            mhw['intensity_cumulative'][ev] = -1.*mhw['intensity_cumulative'][ev]
            mhw['intensity_max_relThresh'][ev] = -1.*mhw['intensity_max_relThresh'][ev]
            mhw['intensity_mean_relThresh'][ev] = -1.*mhw['intensity_mean_relThresh'][ev]
            mhw['intensity_cumulative_relThresh'][ev] = -1.*mhw['intensity_cumulative_relThresh'][ev]
            mhw['intensity_max_abs'][ev] = -1.*mhw['intensity_max_abs'][ev]
            mhw['intensity_mean_abs'][ev] = -1.*mhw['intensity_mean_abs'][ev]
            mhw['intensity_cumulative_abs'][ev] = -1.*mhw['intensity_cumulative_abs'][ev]


    return result,mhw

def runavg(ts, w):
    '''

    Performs a running average of an input time series using uniform window
    of width w. This function assumes that the input time series is periodic.

    Inputs:

      ts            Time series [1D numpy array]
      w             Integer length (must be odd) of running average window

    Outputs:

      ts_smooth     Smoothed time series

    Written by Eric Oliver, Institue for Marine and Antarctic Studies, University of Tasmania, Feb-Mar 2015

    '''
    # Original length of ts
    N = len(ts)
    # make ts three-fold periodic
    ts = np.append(ts, np.append(ts, ts))
    # smooth by convolution with a window of equal weights
    ts_smooth = np.convolve(ts, np.ones(w)/w, mode='same')
    # Only output central section, of length equal to the original length of ts
    ts = ts_smooth[N:2*N]

    return ts