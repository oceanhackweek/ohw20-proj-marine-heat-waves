# Marine Heat Wave (MHW) analysis with xarray (ohw20-proj-marine-heat-waves)

This project aims to apply the MHW definition of Hobday et al. (2016) using the capabilities of [xarray](https://xarray.pydata.org/en/stable/) and dask. The project was started as an Ocean Hack Week 2020 project with the following objectives:

This project objectives are to:
- update [E. Oliver's Marine Heat Waves (MHW) python code](https://github.com/ecjoliver/marineHeatWaves) to take advantage of xarray, datetime, and dask
- adapt to use AWS MUR SST data
- expand to regions (2D) analysis, as well as time series

First steps are to update/rewrite the MHW code using xarray with testing to verify the code produces the same results as [this previous python package](https://github.com/ecjoliver/marineHeatWaves). 

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

_MHW defintions to be here..._


### References

Hobday, A.J., Alexander, L.V., Perkins, S.E., Smale, D.A., Straub, S.C., Oliver, E.C., Benthuysen, J.A., Burrows, M.T., Donat, M.G., Feng, M. and Holbrook, N.J., 2016. A hierarchical approach to defining marine heatwaves. _Progress in Oceanography_, 141, pp.227-238. https://doi.org/10.1016/j.pocean.2015.12.014
