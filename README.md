# Marine Heat Wave (MHW) analysis with xarray (ohw20-proj-marine-heat-waves)

This project aims to apply the MHW definition of Hobday et al. (2016) using the capabilities of [xarray](https://xarray.pydata.org/en/stable/) and [dask](https://dask.org/). The project was started as an Ocean Hack Week 2020 project with the following objectives:

This project objectives are to:
- update [E. Oliver's Marine Heat Waves (MHW) python code](https://github.com/ecjoliver/marineHeatWaves) to take advantage of xarray, datetime, and dask
- adapt to use AWS MUR SST data
- expand to regions (2D) analysis, as well as time series

First steps are to update/rewrite the MHW code using xarray with testing to verify the code produces the same results as [the marineHeatWaves python package](https://github.com/ecjoliver/marineHeatWaves). This version of the python code does the analysis of a single point time series. Here we want to expand to the spatial data available from satellite sea surface temperature (SST) data products, and so changing from the data format in the previous code to one using xarray is the logical step. 

#### Other MHW codes 
In addition to the python [MHW code](https://github.com/ecjoliver/marineHeatWaves), an [R MHW code](https://cran.r-project.org/web/packages/RmarineHeatWaves/index.html) and [Matlab MHW toolbox](https://github.com/ZijieZhaoMMHW/m_mhw1.0) exist. 

### How to contribute? 

Current [issues can be found or created here](https://github.com/oceanhackweek/ohw20-proj-marine-heat-waves/issues). 

### Current status

The Jupyter notebook  Test_Run_Timeseries.ipynb integrates our efforts at the end of Ocean Hack Week 2020. See issues above for more updated status.



### Cloud computing for this project:


To use GCP cloud 'Pangeo Binder GCE us-central1'.

| [![badge](https://img.shields.io/static/v1.svg?logo=Jupyter&label=Pangeo+Binder&message=GCE+us-central1&color=blue)](https://binder.pangeo.io/v2/gh/oceanhackweek/ohw20-proj-marine-heat-waves/master?urlpath=git-pull?repo=https://github.com/oceanhackweek/ohw20-proj-marine-heat-waves%26amp%3Burlpath=lab/tree/ohw20-proj-marine-heat-waves) |

To use AWS cloud 'Pangeo Binder AWS us-west-2'

[![badge](https://img.shields.io/static/v1.svg?logo=Jupyter&label=Pangeo+Binder&message=AWS+us-west-2&color=orange)](https://aws-uswest2-binder.pangeo.io/v2/gh/oceanhackweek/ohw20-proj-marine-heat-waves/master?urlpath=git-pull?repo=https://github.com/oceanhackweek/ohw20-proj-marine-heat-waves%26amp%3Burlpath=lab/tree/ohw20-proj-marine-heat-waves)

### Definitions

The definitions of Hobday et al. (2016) are given below (or refer to Table 2 of the paper). 

*Climatology* 
_T<sub>m</sub>_ (deg C) is the climatological mean, defined with the following equation
<img src="https://render.githubusercontent.com/render/math?math=T_m\left(j\right)=\sum_{y=y_s}^{y_e}\sum_{d=j-5}^{j+5}\frac{T\left(y,d\right)}{11\left(y_e-y_s+1\right)}">. Subscripts _s_ and _e_ are _start_ and _end_ of the climatological record, _T_ is the temperature (defined as daily temperature), _j_ the day of the year index. _T(d,y)_ indicates the SST on the _d:day_ and _y:year_.

*Threshold*
The threshold temperature, _T<sub>%</sub>_ (deg C), is determined by the percentile value desired (i.e. _T<sub>90</sub>_ is 90% threshold, _T<sub>95</sub>_ is a 95% threshold, etc.). The equation is given by <img src="https://render.githubusercontent.com/render/math?math=T_{90}\left(j\right)=P_{90}\left(X\right)"> where _P<sub>90</sub>_ is the 90% and <img src="https://render.githubusercontent.com/render/math?math=X=\left\{T\left(y,d\right)|y_s\leq{y}\leq{y_e},j-5\leq{d}\leq{j+5}\right\}">. Hobday et al. (2016) use the 90th percentile as the threshold to define an MHW.  

*MHW Start and End* 
The start time _t<sub>s</sub>_ (days) and end time _t<sub>e</sub>_ (days) of the MHW are defined when (_start_): T(t)>T<sub>90</sub>(j) and T(t-1)<T<sub>90</sub>(j) and (_end_): t<sub>e</sub>>t<sub>s</sub> and T(t)<T<sub>90</sub>(j) and T(t-1)>T<sub>90</sub>(j)


### References

Hobday, A.J., Alexander, L.V., Perkins, S.E., Smale, D.A., Straub, S.C., Oliver, E.C., Benthuysen, J.A., Burrows, M.T., Donat, M.G., Feng, M. and Holbrook, N.J., 2016. A hierarchical approach to defining marine heatwaves. _Progress in Oceanography_, 141, pp.227-238. https://doi.org/10.1016/j.pocean.2015.12.014
