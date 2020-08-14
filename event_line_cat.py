def event_line_cat(ev, sst, t, mhws, clim):
    
    '''
    Function for plot categories MHW
    Bruna Alves, 2020
    Ocean Hack Week MHW project
    '''

    '''
    inputs:
    'ev'    event number, you must select event from mhws (obtained from mhw.detect)
    'sst'   sea surface temperature variable (1D)
    't'     time
    'mhws'  output from mhw.detect
    'clim'  output from mhw.detect 
    '''
    
    from matplotlib import colors as mcolors
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
    
    dates = [date.fromordinal(tt.astype(int)) for tt in t]
    diff = clim['thresh']-clim['seas']     
    threshold2x = clim['thresh']+diff
    threshold3x = threshold2x+diff
    threshold4x = threshold3x+diff
    
    if ev>2:
        evi=ev-2
    else:
        evi=0
        
    plt.figure(figsize=(16,8))        
    for ev0 in np.arange(evi, ev+2, 1):
        t1 = np.where(t==mhws['time_start'][ev0])[0][0]
        t2 = np.where(t==mhws['time_end'][ev0])[0][0]
    
        #Plot fill
        #Moderate
        plt.fill_between(dates[t1:t2+1], sst[t1:t2+1], clim['thresh'][t1:t2+1], color=colors['gold'])
    
        #Strong
        plt.fill_between(dates[t1:t2+1], sst[t1:t2+1], threshold2x[t1:t2+1],
                  where = (sst[t1:t2+1] > clim['thresh'][t1:t2+1]) & (sst [t1:t2+1] > threshold2x[t1:t2+1]), 
                  color=colors['coral'])
    
        #Severe
        plt.fill_between(dates[t1:t2+1], sst[t1:t2+1], threshold3x[t1:t2+1], 
                  where = (sst[t1:t2+1] > threshold2x[t1:t2+1]) & (sst [t1:t2+1] > threshold3x[t1:t2+1]), 
                  color=colors['crimson'])
    
        #Extreme
        plt.fill_between(dates[t1:t2+1], sst[t1:t2+1], threshold4x[t1:t2+1], 
                     where = (sst[t1:t2+1] > threshold3x[t1:t2+1]) & (sst [t1:t2+1] > threshold4x[t1:t2+1]), 
                     color=colors['darkred'])

    
        # Plot SST, Thresh, 2x, 3x, 4x
    plt.plot(dates, sst, 'k-', linewidth=2,label='SST')
    plt.plot(dates, clim['seas'], '-', linewidth=2, color=colors['steelblue'],label = 'Climatology')
    plt.plot(dates, clim['thresh'], 'g-', linewidth=2, label = 'Threshold')
    plt.plot(dates,threshold2x, 'g--',linewidth=2,label = '2x Threshold')
    plt.plot(dates,threshold3x, 'g-.',linewidth=2, label = '3x Threshold')
    plt.plot(dates,threshold4x, 'g:',linewidth=2, label = '4x Threshold')
    plt.title('MHW Categories',fontdict={'fontsize': 14, 'fontweight': 'bold'})       
    plt.xlim(datetime.date.fromordinal(mhws['time_start'][ev]-150), datetime.date.fromordinal(mhws['time_end'][ev]+150))
    plt.legend(prop={"size":12})
    plt.xlabel('Time',fontdict={'fontsize': 14})
    plt.ylabel('SST',fontdict={'fontsize': 14})
