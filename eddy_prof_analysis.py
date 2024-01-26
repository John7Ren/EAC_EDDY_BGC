import xarray as xr
import numpy as np
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import pandas as pd
import matplotlib as mpl
from matplotlib import path
from scipy import interpolate
import warnings
from glob import glob
import os
import matplotlib.gridspec as gridspec
from geopy import distance
import geopy as gp
from matplotlib.animation import FuncAnimation
import matplotlib.collections as clt
import matplotlib.animation as animation
import gsw
from scipy.io import loadmat
import mat73

# Function
# ----------------------------------- Common -----------------------------------
def addParas(prof,eddy):
    eddy_exist = len(list(eddy.dims.keys()))

    month = np.array(pd.Series(prof.time.values).dt.month)
    season = np.zeros_like(month)
    sls = np.array([[12,1, 2],
                    [3, 4, 5],
                    [6, 7, 8],
                    [9, 10,11]])
    for i in range(sls.shape[0]):
        season[np.in1d(month,sls[i,:])] = i

    if eddy_exist > 0:
        longevity = np.ones_like(prof.n_prof.values).astype('float')
        lat_birth = np.ones_like(prof.n_prof.values).astype('float')
        lon_birth = np.ones_like(prof.n_prof.values).astype('float')
        time_birth = np.ones_like(prof.n_prof.values).astype('float')
        area = np.ones_like(prof.n_prof.values).astype('float')
        amplitude = np.ones_like(prof.n_prof.values).astype('float')
        iage= np.ones_like(prof.n_prof.values).astype('float')

        for i in range(len(prof.n_prof.values)):
            ied = prof.eddyidx.values[i]
            mask = eddy['eddyidx_obs'].values == ied
            r,_ = np.where(mask)
            longevity[i] = eddy.sel(n_eddy=r)['longevity']
            lat_birth[i] = eddy.sel(n_eddy=r)['lat_birth']
            lon_birth[i] = eddy.sel(n_eddy=r)['lon_birth']
            time_birth[i] = eddy.sel(n_eddy=r)['time_birth']
            area[i] = eddy.sel(n_eddy=r)['area_t']
            amplitude[i] = eddy.sel(n_eddy=r)['amplitude_t']
            iage[i] = eddy['age'].values[mask] / longevity[i]

        prof = prof.assign_coords({'month':(['n_prof'],month),
                                    'season':(['n_prof'],season),
                                    'longevity':(['n_prof'],longevity),
                                    'lat_birth':(['n_prof'],lat_birth),
                                    'lon_birth':(['n_prof'],lon_birth),
                                    'time_birth':(['n_prof'],time_birth),
                                    'area':(['n_prof'],area),
                                    'amplitude':(['n_prof'],amplitude),
                                    'iage':(['n_prof'],iage)})
    else:
        prof = prof.assign_coords({'month':(['n_prof'],month),
                                   'season':(['n_prof'],season)})
        
    return prof

def combineEDDY(ds1,ds2):
    track1 = ds1.track.values
    track2 = ds2.track.values
    ds2 = ds2.sel(n_eddy=~np.in1d( track2,track1 ))
    l1 = ds1.eddyidx_obs.values.shape[1]
    l2 = ds2.eddyidx_obs.values.shape[1]
    dl = l1-l2
    if dl < 0:
        dl = abs(dl)
        ds1 = ds1.pad(pad_width={'obs': (0,dl)},mode='constant',constant_values={'obs': (0,-999)})
    elif dl > 0:
        ds2 = ds2.pad(pad_width={'obs': (0,dl)},mode='constant',constant_values={'obs': (0,-999)})
    DS = xr.concat([ds1,ds2],dim='n_eddy')
    return DS

def ieddy_ds(ied,eddy):
    eddyidx_obs = np.arange(0,len(eddy.amplitude.values))
    eddy = eddy.assign_coords({'eddyidx_obs':(['obs'],eddyidx_obs)})
    track = eddy.isel(obs=ied).track.values
    keys = ['amplitude','time','effective_area','latitude','longitude']
    eddy1 = eddy.sel(obs=eddy.track==track)[keys]

    # coords
    eddyidx_obs = eddy1['eddyidx_obs'].values
    l = len(eddyidx_obs)
    # data_vars, 2D
    amplitude = eddy1['amplitude'].values * 0.0001
    area = eddy1['effective_area'].values
    time = days2dt64(eddy1['time'].values)
    age = time - min(time)
    lon = eddy1.longitude.values
    lat = eddy1.latitude.values
    # store time datetime64[ns] into int for padding
    time = np.array(time).astype(int)
    age = np.array(age.dt.days)
    # data_vars, 1D
    amplitude_t = np.round(max(amplitude),4)
    area_t = max(area)
    time_birth = min(time)
    lat_birth = min(lat)
    lon_birth = max(lon)
    longevity = max(age)

    ds = xr.Dataset(
        data_vars = {
                    #  2D vars
                    'amplitude': ( ['n_eddy','obs'],amplitude.reshape(1,l) ),
                    'area':      ( ['n_eddy','obs'],area.reshape(1,l) ),
                    'time':      ( ['n_eddy','obs'],np.array(time).reshape(1,l) ),
                    'age':       ( ['n_eddy','obs'],np.array(age).reshape(1,l) ),
                    'lon':       ( ['n_eddy','obs'],lon.reshape(1,l) ),
                    'lat':       ( ['n_eddy','obs'],lat.reshape(1,l) ),
                    #  1D vars
                    'amplitude_t':( ['n_eddy'],[amplitude_t] ),
                    'area_t':     ( ['n_eddy'],[area_t] ),
                    'time_birth': ( ['n_eddy'],[time_birth] ),
                    'longevity':  ( ['n_eddy'],[longevity] ),
                    'lat_birth':  ( ['n_eddy'],[lat_birth] ),
                    'lon_birth':  ( ['n_eddy'],[lon_birth] ),
                    'track':      ( ['n_eddy'],[track] )},
        
        coords = {'eddyidx_obs' : ( ['n_eddy','obs'],eddyidx_obs.reshape(1,l) )}
    )
    return ds

# def makeEddy

def inpaintnan1d(var_array,pres_old,res):
    var_array = var_array.flatten()
    pres_old = pres_old.flatten()
    # interpolate pressure
    pres0 = np.floor(np.nanmin(pres_old))
    pres1 = np.ceil(np.nanmax(pres_old)+1)
    pres_itp = np.arange(pres0,pres1+res/2,res)
    # remove nan and infinite
    mask_valid = np.isfinite(var_array) & np.isfinite(pres_old)
    # count the in-situ observation number and assign them to the interpolated pressure bins
    count, _ = np.histogram(pres_old[mask_valid],pres_itp)
    count_pad = np.pad(count,pad_width=(0,1),mode='constant',constant_values=0)
    if mask_valid.sum() > 1:
        variable_valid = var_array[mask_valid]
        pres_old_valid = pres_old[mask_valid]
        f = interpolate.interp1d(pres_old_valid,variable_valid,bounds_error=False,fill_value='extrapolate')
        var_itp = f(pres_itp)
    else:
        var_itp = np.nan * np.ones_like(pres_itp)
    return var_itp,pres_itp,count_pad

def addSigma0(prof):
    # load values
    lon = prof['lon'].values
    lat = prof['lat'].values
    T = prof['temperature'].values
    SP = prof['salinity'].values
    p = prof['pressure'].values

    # reshape
    P = p.reshape(len(p),1)
    LON = lon.reshape(1,len(lon))
    LAT = lat.reshape(1,len(lat))

    # tile
    P = np.tile(P,(1,len(lon)))
    LON = np.tile(LON,(len(p),1))
    LAT = np.tile(LAT,(len(p),1))

    # calculate
    SA = gsw.conversions.SA_from_SP(SP,P,LON,LAT)
    CT = gsw.conversions.CT_from_t(SA,T,P)

    prof = prof.assign({'sigma0':(['pressure','n_prof'],gsw.density.sigma0(SA,CT))})
    return prof

def findperipheral(ilon_cars,ilat_cars,ds_cars):
    lons = ds_cars['lon'].values
    lats = ds_cars['lat'].values
    idx_lon = np.argwhere(np.isclose(lons,ilon_cars))[0][0] # index the value from an array
    idx_lat = np.argwhere(np.isclose(lats,ilat_cars))[0][0] # index the value from an array
    lons_sel = lons[idx_lon-1:idx_lon+2]
    lats_sel = lats[idx_lat-1:idx_lat+2]
    lons_mesh,lats_mesh = np.meshgrid(lons_sel,lats_sel)
    return lons_mesh.flatten(order='C'),lats_mesh.flatten(order='C')

def iCARS(time,ilon,ilat,pressure,ds_cars):
    # get the peripheral grid dots
    ilon_1dot = ds_cars.sel({'lon':ilon,'lat':ilat},method='nearest')['lon'].values
    ilat_1dot = ds_cars.sel({'lon':ilon,'lat':ilat},method='nearest')['lat'].values
    ilon_9dots,ilat_9dots = findperipheral(ilon_1dot,ilat_1dot,ds_cars)

    # get the climatology data at 1 dot or 9 dots
    lons = ilon_9dots
    lats = ilat_9dots
    dat_mean = ds_cars.sel({'lon':lons,'lat':lats},method='nearest')['mean'].values
    an_cos = ds_cars.sel({'lon':lons,'lat':lats},method='nearest')['an_cos'].values
    an_sin = ds_cars.sel({'lon':lons,'lat':lats},method='nearest')['an_sin'].values
    sa_cos = ds_cars.sel({'lon':lons,'lat':lats},method='nearest')['sa_cos'].values
    sa_sin = ds_cars.sel({'lon':lons,'lat':lats},method='nearest')['sa_sin'].values
    depth = ds_cars['depth'].values

    # reshape for matrix multiplication
    dat_mean = dat_mean.reshape(len(depth),len(lats),len(lons),1)
    depth = depth
    an_cos_ext = np.zeros((len(depth),len(lats),len(lons),1))
    an_sin_ext = np.zeros((len(depth),len(lats),len(lons),1))
    sa_cos_ext = np.zeros((len(depth),len(lats),len(lons),1))
    sa_sin_ext = np.zeros((len(depth),len(lats),len(lons),1))
    an_cos_ext[:len(an_cos),:,:,0] = an_cos
    an_sin_ext[:len(an_sin),:,:,0] = an_sin
    sa_cos_ext[:len(sa_cos),:,:,0] = sa_cos
    sa_sin_ext[:len(sa_sin),:,:,0] = sa_sin

    t0 = time.astype('datetime64[D]')-15
    t1 = time.astype('datetime64[D]')+16
    time_range = np.arange(t0,t1)
    data_shape = dat_mean.shape[:3] + (len(time_range),) # add 1 time dimension to the data -> (depth,lat,lon,time)

    time_range = time_range.reshape(1,1,1,data_shape[3])
    t_range = 2*np.pi * (time_range - np.datetime64('2023-01-01'))/np.timedelta64(366,'D') # be aware that this only works for date that after 2023-01-16, and before 2023-12-15

    # tile the data
    mean_tile = np.tile(dat_mean,(1,1,1)+(data_shape[3],))
    t_tile = np.tile(t_range,data_shape[:3]+(1,))
    an_cos_tile = np.tile(an_cos_ext,(1,1,1)+(data_shape[3],))
    an_sin_tile = np.tile(an_sin_ext,(1,1,1)+(data_shape[3],))
    sa_cos_tile = np.tile(sa_cos_ext,(1,1,1)+(data_shape[3],))
    sa_sin_tile = np.tile(sa_sin_ext,(1,1,1)+(data_shape[3],))

    # calculation
    climat_range = mean_tile + an_cos_tile*np.cos(t_tile) + an_sin_tile*np.sin(t_tile) + sa_cos_tile*np.cos(2*t_tile) + sa_sin_tile*np.sin(2*t_tile)

    # make the negative value to zero if there was any
    if climat_range.any() < 0:
        print(f'on {time}, at (lon,lat) ({ilon},{ilat})')
    climat_range = np.where(climat_range<0,0,climat_range)

    # take average
    iclimat_mean = np.nanmean(climat_range,axis=(1,2,3))

    # interpolation
    f = interpolate.interp1d(depth,iclimat_mean,bounds_error=False,fill_value='extrapolate')
    iclimat_itp = f(pressure)
    
    return iclimat_itp,lons,lats

def addCARS(prof,prof_type):
    # lon_arr = np.zeros((9,len(prof.n_prof.values)))
    # lat_arr = np.zeros((9,len(prof.n_prof.values)))
    if prof_type == 'ctd':
        var_list = ['temperature','salinity','oxygen']
    elif prof_type == 'hydro':
        var_list = ['nitrate','phosphate','silicate']
    elif prof_type == 'argo':
        var_list = ['temperature','salinity','oxygen','nitrate']
    elif prof_type == 'v06' or prof_type == 'his':
        var_list = ['temperature','salinity','oxygen','nitrate','phosphate','silicate']
    else:
        print('invalid inputs')
    dir = '/Users/renjiongqiu/Library/CloudStorage/OneDrive-UniversityofTasmania/Documents/eddy_tracking/CARS/'
    for var in (var_list):
        fn = (glob(f"{dir}{var}*"))
        ds_cars = xr.open_dataset(fn[0])
        for id in range(len(prof.n_prof.values)):
            # get the ctd data at id
            ilon = prof.lon[id].values
            ilat = prof.lat[id].values
            iprof = prof.n_prof[id].values
            # return prof,lon_arr,lat_arr
            # print(f'id={id}')
            # print(f'{prof.x}')
            
            time = prof.time[id].values
            pressure = prof.pressure.values

            iclimat_itp,lons,lats = iCARS(time,ilon,ilat,pressure,ds_cars)
            # lon_arr[:,id] = lons
            # lat_arr[:,id] = lats
            if var == 'oxygen':
                iclimat_itp = iclimat_itp * 44.661
            da = xr.DataArray(data=iclimat_itp.reshape(len(pressure),1),
                              dims=['pressure','n_prof'],
                              coords={'n_prof':(['n_prof'],[iprof]),
                                      'pressure':pressure})
            ds = da.to_dataset(name=f'{var}_cars')
            ds = ds.drop_duplicates(dim=...)
            # print(ds)
            prof = xr.merge([prof,ds])

    # return prof,lon_arr,lat_arr
    return prof

def calculateAnom(prof,prof_type):
    var_list = list(prof.data_vars.keys())
    var_ar = np.array(var_list)
    if prof_type == 'ctd':
        var_choice = ['temperature','salinity','oxygen']
    elif prof_type == 'hydro':
        var_choice = ['nitrate','phosphate','silicate']
    elif prof_type == 'argo':
        var_choice = ['temperature','salinity','oxygen','nitrate']
    elif prof_type == 'v06' or prof_type == 'his':
        var_choice = ['temperature','salinity','oxygen','nitrate','phosphate','silicate']
    else:
        print('invalid inputs')
    var_cross = var_ar[np.in1d(var_ar,var_choice)]
    for var in var_cross:
        prof[var+'_anom'] = prof[var] - prof[var+'_cars']
    return prof

def makeProf(prof_type,eddy_type,test=False):
    if eddy_type == 'C' or eddy_type == 'A':
        if prof_type == 'argo':
            prof,eddy = makeArgoProf(eddy_type,test=test)
        elif prof_type == 'his':
            prof,eddy = makeHisProf(eddy_type,test=test)
        elif prof_type == 'v06':
            prof,eddy = makeV06Prof(eddy_type)
        else:
            print('invalid profile type')
    elif eddy_type == 'O':
        if prof_type == 'argo':
            prof,eddy = makeArgoProf_OUT(eddy_type,test=test)
        elif prof_type == 'his':
            prof,eddy = makeHisProf_OUT(eddy_type,test=test)
        elif prof_type == 'v06':
            prof,eddy = makeV06Prof_OUT(eddy_type)
        else:
            print('invalid profile type')
    return prof,eddy
# ----------------------------------- Argo -----------------------------------
def days2dt64(time):
    days = time*1.1574074074074073e-05*timedelta(days=1)
    dt = datetime(1950,1,1)+days
    dt64 = pd.Series(dt,dtype='datetime64[ns]')
    return dt64

def argoJULD2TS(argoJULD):
    day0_ts = pd.Timestamp('1950-01-01T00:00:00')
    day0_juld = pd.Timestamp.to_julian_date(day0_ts)
    TS = pd.to_datetime(argoJULD+day0_juld,origin='julian',unit='D')
    return TS

def cycidx2num():
    dir = '/Users/renjiongqiu/Library/CloudStorage/OneDrive-UniversityofTasmania/Documents/eddy_tracking/data/'
    argodata_mat = mat73.loadmat(dir+'argo_data.mat')
    argoineddy_mat = mat73.loadmat(dir+'argoineddy.mat')

    argoineddy = argoineddy_mat['argoineddy']
    
    cycidices = argoineddy['argoindex'][:,1].astype(int)
    floatidices = argoineddy['argoindex'][:,0].astype(int)
    cycidx = np.ones_like(floatidices).astype(int)
    dir = '/Users/renjiongqiu/Library/CloudStorage/OneDrive-UniversityofTasmania/Documents/aa_in2023_v06/Profiles/'
    for icyc,cyc in enumerate(cycidices):
        fl = floatidices[icyc]
        fn = glob(dir+f'{fl}*.nc')
        ds = xr.open_dataset(fn[0])
        cycidx[icyc] = ds['CYCLE_NUMBER'][cyc-1] # the cyc was in MATLAB
    argoineddy_new = np.concatenate([floatidices,cycidx],axis=1)
    np.save('/Users/renjiongqiu/Library/CloudStorage/OneDrive-UniversityofTasmania/Documents/eddy_tracking/data/argoineddy_new.npy',argoineddy_new)

def changeName(prof):
    argoName_selected = np.array(['TEMP_ADJUSTED','PSAL_ADJUSTED','DOXY_ADJUSTED','NITRATE_ADJUSTED','CHLA_ADJUSTED','CHLA_FLUORESCENCE_ADJUSTED',
                                  'TEMP_ADJUSTED_obs','PSAL_ADJUSTED_obs','DOXY_ADJUSTED_obs','NITRATE_ADJUSTED_obs','CHLA_ADJUSTED_obs','CHLA_FLUORESCENCE_ADJUSTED_obs'])
    argoName_ar = np.array(list(prof.data_vars.keys()))
    argoName_cross = argoName_ar[np.in1d(argoName_ar,argoName_selected)]
    newName = {} 
    newName['TEMP_ADJUSTED'] = 'temperature'
    newName['PSAL_ADJUSTED'] = 'salinity'
    newName['DOXY_ADJUSTED'] = 'oxygen'
    newName['NITRATE_ADJUSTED'] = 'nitrate'
    newName['CHLA_ADJUSTED'] = 'fluorescence'
    newName['CHLA_FLUORESCENCE_ADJUSTED'] = 'fluorescence'

    newName['TEMP_ADJUSTED_obs'] = 'temperature_obs'
    newName['PSAL_ADJUSTED_obs'] = 'salinity_obs'
    newName['DOXY_ADJUSTED_obs'] = 'oxygen_obs'
    newName['NITRATE_ADJUSTED_obs'] = 'nitrate_obs'
    newName['CHLA_ADJUSTED_obs'] = 'fluorescence_obs'
    newName['CHLA_FLUORESCENCE_ADJUSTED_obs'] = 'fluorescence_obs'
    for old in argoName_cross:
        new = newName[old]
        prof[new] = prof[old]
        prof = prof.drop(old)
    return prof

def useADJUSTED(var_all,var_list):
    var_adjusted = []
    var_cross = var_all[np.in1d(var_all,var_list)]
    for var in var_cross:
        if var+'_ADJUSTED' in var_all:
            var_adjusted.append(var+'_ADJUSTED')
        else:
            var_adjusted.append(var)
            print(f'no adjusted for {var}')
    return var_adjusted


def makeArgoProf(eddy_type,changename=True,test=False):
    var_list = ['TEMP','PSAL','DOXY','CHLA','NITRATE','CHLA_FLUORESCENCE']
    dir = '/Users/renjiongqiu/Library/CloudStorage/OneDrive-UniversityofTasmania/Documents/eddy_tracking/data/'
    argodata_mat = mat73.loadmat(dir+'argo_data.mat')
    argoineddy_mat = mat73.loadmat(dir+'argoineddy.mat')

    argoineddy = argoineddy_mat['argoineddy']
    argodata = argodata_mat['Data']

    # create masks
    eddytype_split = np.array(list(argoineddy['eddytype'][:4121]))
    mask_edy = (eddytype_split==eddy_type)

    # extract index
    eddyidx = argoineddy['eddyindex'][mask_edy].astype(int)
    floatidx = argoineddy['argoindex'][mask_edy,0].astype(int)
    cycidx = np.load('/Users/renjiongqiu/Library/CloudStorage/OneDrive-UniversityofTasmania/Documents/eddy_tracking/data/argoineddy_new.npy')[mask_edy,1]
    if test:
        l = 5
    else:
        l = len(eddyidx)

    # import eddy
    if eddy_type == 'C':
        eddy = xr.open_dataset('/Users/renjiongqiu/Library/CloudStorage/OneDrive-UniversityofTasmania/Documents/eddy_tracking/cyclonic_eddy_traj_eac_days_more30_meta3p2_dt_2sat.nc', decode_cf=False)
    elif eddy_type == 'A':
        eddy = xr.open_dataset('/Users/renjiongqiu/Library/CloudStorage/OneDrive-UniversityofTasmania/Documents/eddy_tracking/anticyclonic_eddy_traj_eac_days_more30_meta3p2_dt_2sat.nc', decode_cf=False)
    else:
        print('invalid input')
    eddy['effective_radius'] = eddy['effective_radius'].astype(float)

    for i in range(l):
        ied = eddyidx[i]-1
        ifl = floatidx[i]
        icyc = cycidx[i]

        # make profile dataset
        ifloat = argodata[f'F{ifl}']
        var_all = np.array(list(ifloat.keys()))
        var_adjusted = useADJUSTED(var_all,var_list)
        mask_cyc = (ifloat['CYCLE_NUMBER']==icyc)
        lat = np.mean(ifloat['LATITUDE'][mask_cyc])
        lon = np.mean(ifloat['LONGITUDE'][mask_cyc])
        coords = np.array([lat,lon])
        juld = np.mean(ifloat['JULD'][mask_cyc])
        time = argoJULD2TS(juld)
        pres_name = useADJUSTED(var_all,['PRES'])[0]
        pres_old = ifloat[pres_name][mask_cyc].flatten()
        pres_valid = pres_old[np.isfinite(pres_old)]
        
        lon_edd = eddy.isel(obs=ied).longitude.values
        lat_edd = eddy.isel(obs=ied).latitude.values
        lon_edd_eff = eddy.isel(obs=ied).longitude_max.values
        lat_edd_eff = eddy.isel(obs=ied).latitude_max.values
        rad = eddy.isel(obs=ied).effective_radius.values * 50
        rad_deg = rad / 1000 * 0.009
        lon_cont = eddy.isel(obs=ied).effective_contour_longitude.values.reshape(1,20)*0.01+180
        lat_cont = eddy.isel(obs=ied).effective_contour_latitude.values.reshape(1,20)*0.01
        time_edd = days2dt64(eddy.isel(obs=ied).time.values)[0]

        # index the corresponding eddy and calculate x
        coords_edd = np.array([lat_edd,lon_edd])
        # coords_edd = np.array([lat_edd[i,0],lon_edd[i,0]])
        dist = distance.distance(coords_edd,coords).m
        x = dist/rad

        # extract vars and interpolate when pressure level is larger than 1
        if len(pres_valid) > 1:
            ds = xr.Dataset({})
            for ivar,var in enumerate(var_adjusted):
                res = 1
                var_array = ifloat[var][mask_cyc]
                var_itp,pres_itp,obs_n = inpaintnan1d(var_array,pres_old,res)
                var_itp = var_itp.reshape(len(var_itp),1)
                obs_n = obs_n.reshape(len(obs_n),1)
                ds[var] = xr.DataArray(data=var_itp,
                                dims=['pressure','n_prof'],
                                coords={'pressure':(['pressure'],pres_itp),
                                        'validity':(['n_prof'],['yes']),
                                        'n_prof':(['n_prof'],[i]),
                                        'eddyidx':(['n_prof'],[ied]),
                                        'float':(['n_prof'],[ifl]),
                                        'cycle_number':(['n_prof'],[icyc]),
                                        'x':(['n_prof'],[x]),
                                        'lon':(['n_prof'],[lon]),
                                        'lat':(['n_prof'],[lat]),
                                        'time':(['n_prof'],[time]),
                                        'time_edd':(['n_prof'],[time_edd]),
                                        'lon_edd':(['n_prof'],[lon_edd]),
                                        'lat_edd':(['n_prof'],[lat_edd]),
                                        'lon_edd_eff':(['n_prof'],[lon_edd_eff]),
                                        'lat_edd_eff':(['n_prof'],[lat_edd_eff]),
                                        'rad':(['n_prof'],[rad]),
                                        'rad_deg':(['n_prof'],[rad_deg])})
                ds[var+'_obs'] = xr.DataArray(data=obs_n,
                                dims=['pressure','n_prof'],
                                coords={'pressure':(['pressure'],pres_itp),
                                        'validity':(['n_prof'],['yes']),
                                        'n_prof':(['n_prof'],[i]),
                                        'eddyidx':(['n_prof'],[ied]),
                                        'float':(['n_prof'],[ifl]),
                                        'cycle_number':(['n_prof'],[icyc]),
                                        'x':(['n_prof'],[x]),
                                        'lon':(['n_prof'],[lon]),
                                        'lat':(['n_prof'],[lat]),
                                        'time':(['n_prof'],[time]),
                                        'time_edd':(['n_prof'],[time_edd]),
                                        'lon_edd':(['n_prof'],[lon_edd]),
                                        'lat_edd':(['n_prof'],[lat_edd]),
                                        'lon_edd_eff':(['n_prof'],[lon_edd_eff]),
                                        'lat_edd_eff':(['n_prof'],[lat_edd_eff]),
                                        'rad':(['n_prof'],[rad]),
                                        'rad_deg':(['n_prof'],[rad_deg])})
            ds = ds.assign_coords({'lon_cont':(['n_prof','Ncont'],lon_cont.data),'lat_cont':(['n_prof','Ncont'],lat_cont.data)})
        elif len(pres_valid) == 1:
            ds = xr.Dataset({})
            for ivar,var in enumerate(var_adjusted):
                ds[var] = xr.DataArray(data=ifloat[var][mask_cyc].reshape(1,1),
                                dims=['pressure','n_prof'],
                                coords={'pressure':(['pressure'],[0]),
                                        'validity':(['n_prof'],['no']),
                                        'n_prof':(['n_prof'],[i]),
                                        'eddyidx':(['n_prof'],[ied]),
                                        'float':(['n_prof'],[ifl]),
                                        'cycle_number':(['n_prof'],[icyc]),
                                        'x':(['n_prof'],[x]),
                                        'lon':(['n_prof'],[lon]),
                                        'lat':(['n_prof'],[lat]),
                                        'time':(['n_prof'],[time]),
                                        'time_edd':(['n_prof'],[time_edd]),
                                        'lon_edd':(['n_prof'],[lon_edd]),
                                        'lat_edd':(['n_prof'],[lat_edd]),
                                        'lon_edd_eff':(['n_prof'],[lon_edd_eff]),
                                        'lat_edd_eff':(['n_prof'],[lat_edd_eff]),
                                        'rad':(['n_prof'],[rad]),
                                        'rad_deg':(['n_prof'],[rad_deg])})
                ds[var+'_obs'] = xr.DataArray(data=[[1]],
                                dims=['pressure','n_prof'],
                                coords={'pressure':(['pressure'],pres_itp),
                                        'validity':(['n_prof'],['no']),
                                        'n_prof':(['n_prof'],[i]),
                                        'eddyidx':(['n_prof'],[ied]),
                                        'float':(['n_prof'],[ifl]),
                                        'cycle_number':(['n_prof'],[icyc]),
                                        'x':(['n_prof'],[x]),
                                        'lon':(['n_prof'],[lon]),
                                        'lat':(['n_prof'],[lat]),
                                        'time':(['n_prof'],[time]),
                                        'time_edd':(['n_prof'],[time_edd]),
                                        'lon_edd':(['n_prof'],[lon_edd]),
                                        'lat_edd':(['n_prof'],[lat_edd]),
                                        'lon_edd_eff':(['n_prof'],[lon_edd_eff]),
                                        'lat_edd_eff':(['n_prof'],[lat_edd_eff]),
                                        'rad':(['n_prof'],[rad]),
                                        'rad_deg':(['n_prof'],[rad_deg])})
            ds = ds.assign_coords({'lon_cont':(['n_prof','Ncont'],lon_cont.data),'lat_cont':(['n_prof','Ncont'],lat_cont.data)})
        else:
            print(f'PRES_ADJUSTED is all nan at F{ifl} - C{icyc}')

        if i == 0:
            DS = ds.copy()
            # create eddy dataset
            ds_ed = ieddy_ds(ied,eddy)
            DS_ED = ds_ed.copy()
        else:
            DS = xr.merge([DS,ds])
            # DS = xr.concat([DS,ds],dim='n_prof')

            track = eddy.isel(obs=ied).track.values
            if track not in DS_ED.track.values:
                ds_ed = ieddy_ds(ied,eddy)
                DS_ED = combineEDDY(DS_ED,ds_ed)
        # DS = DS.sortby('x')
        DS = DS.transpose('pressure','n_prof','Ncont')
        DS_ED = DS_ED.transpose('n_eddy','obs')
        mask = np.isnan(DS_ED.eddyidx_obs.values)
        DS_ED['eddyidx_obs'] = (['n_eddy','obs'],np.where(mask,-999,DS_ED.eddyidx_obs.values.astype(int)))
        for key in ['float','cycle_number','eddyidx']:
            DS[key] = DS[key].astype(int)
    
    if changename:
        DS = changeName(DS)
        
    return DS,DS_ED

def makeArgoProf_OUT(eddy_type='O',changename=True,test=False):
    var_list = ['TEMP','PSAL','DOXY','CHLA','NITRATE','CHLA_FLUORESCENCE']
    dir = '/Users/renjiongqiu/Library/CloudStorage/OneDrive-UniversityofTasmania/Documents/eddy_tracking/data/'
    argodata_mat = mat73.loadmat(dir+'argo_data.mat')
    argoineddy_mat = mat73.loadmat(dir+'argoineddy.mat')

    argoineddy = argoineddy_mat['argoineddy']
    argodata = argodata_mat['Data']

    # create masks
    eddytype_split = np.array(list(argoineddy['eddytype'][:4121]))
    mask_edy = (eddytype_split==eddy_type)

    # extract index
    floatidx = argoineddy['argoindex'][mask_edy,0].astype(int)
    cycidx = np.load('/Users/renjiongqiu/Library/CloudStorage/OneDrive-UniversityofTasmania/Documents/eddy_tracking/data/argoineddy_new.npy')[mask_edy,1]
    if test:
        l = 5
    else:
        l = len(floatidx)

    for i in range(l):
        ifl = floatidx[i]
        icyc = cycidx[i]

        # make profile dataset
        ifloat = argodata[f'F{ifl}']
        var_all = np.array(list(ifloat.keys()))
        var_adjusted = useADJUSTED(var_all,var_list)
        mask_cyc = (ifloat['CYCLE_NUMBER']==icyc)
        lat = np.mean(ifloat['LATITUDE'][mask_cyc])
        lon = np.mean(ifloat['LONGITUDE'][mask_cyc])
        coords = np.array([lat,lon])
        juld = np.mean(ifloat['JULD'][mask_cyc])
        time = argoJULD2TS(juld)
        pres_name = useADJUSTED(var_all,['PRES'])[0]
        pres_old = ifloat[pres_name][mask_cyc].flatten()
        pres_valid = pres_old[np.isfinite(pres_old)]

        # extract vars and interpolate when pressure level is larger than 1
        if len(pres_valid) > 1:
            ds = xr.Dataset({})
            for ivar,var in enumerate(var_adjusted):
                res = 1
                var_array = ifloat[var][mask_cyc]
                var_itp,pres_itp,obs_n = inpaintnan1d(var_array,pres_old,res)
                var_itp = var_itp.reshape(len(var_itp),1)
                obs_n = obs_n.reshape(len(obs_n),1)
                ds[var] = xr.DataArray(data=var_itp,
                                dims=['pressure','n_prof'],
                                coords={'pressure':(['pressure'],pres_itp),
                                        'validity':(['n_prof'],['yes']),
                                        'n_prof':(['n_prof'],[i]),
                                        'float':(['n_prof'],[ifl]),
                                        'cycle_number':(['n_prof'],[icyc]),
                                        'lon':(['n_prof'],[lon]),
                                        'lat':(['n_prof'],[lat]),
                                        'time':(['n_prof'],[time])})
                ds[var+'_obs'] = xr.DataArray(data=obs_n,
                                dims=['pressure','n_prof'],
                                coords={'pressure':(['pressure'],pres_itp),
                                        'validity':(['n_prof'],['yes']),
                                        'n_prof':(['n_prof'],[i]),
                                        'float':(['n_prof'],[ifl]),
                                        'cycle_number':(['n_prof'],[icyc]),
                                        'lon':(['n_prof'],[lon]),
                                        'lat':(['n_prof'],[lat]),
                                        'time':(['n_prof'],[time])})
                if var == 'CHLA_FLUORESCENCE_ADJUSTED':
                    var = 'CHLA_ADJUSTED'

        elif len(pres_valid) == 1:
            ds = xr.Dataset({})
            for ivar,var in enumerate(var_adjusted):
                ds[var] = xr.DataArray(data=ifloat[var][mask_cyc].reshape(1,1),
                                dims=['pressure','n_prof'],
                                coords={'pressure':(['pressure'],[0]),
                                        'validity':(['n_prof'],['no']),
                                        'n_prof':(['n_prof'],[i]),
                                        'float':(['n_prof'],[ifl]),
                                        'cycle_number':(['n_prof'],[icyc]),
                                        'lon':(['n_prof'],[lon]),
                                        'lat':(['n_prof'],[lat]),
                                        'time':(['n_prof'],[time])})
                ds[var+'_obs'] = xr.DataArray(data=[[1]],
                                dims=['pressure','n_prof'],
                                coords={'pressure':(['pressure'],[0]),
                                        'validity':(['n_prof'],['no']),
                                        'n_prof':(['n_prof'],[i]),
                                        'float':(['n_prof'],[ifl]),
                                        'cycle_number':(['n_prof'],[icyc]),
                                        'lon':(['n_prof'],[lon]),
                                        'lat':(['n_prof'],[lat]),
                                        'time':(['n_prof'],[time])})
                if var == 'CHLA_FLUORESCENCE_ADJUSTED':
                    var = 'CHLA_ADJUSTED'
        else:
            print(f'PRES_ADJUSTED is all nan at F{ifl} - C{icyc}')

        if i == 0:
            DS = ds.copy()
        else:
            DS = xr.merge([DS,ds])
            # DS = xr.concat([DS,ds],dim='n_prof')
        # DS = DS.sortby('x')
        DS = DS.transpose('pressure','n_prof')
        for key in ['float','cycle_number']:
            DS[key] = DS[key].astype(int)
    
    if changename:
        DS = changeName(DS)
    else:
        print('have to change var name to proceed the following processes: addSigma0, addCARS, calculateAnom')
    DS_ED = xr.Dataset({})
    return DS,DS_ED
# ----------------------------------- History -----------------------------------

def importIndices(ctd_type,eddy_type):
    dir = '/Users/renjiongqiu/Library/CloudStorage/OneDrive-UniversityofTasmania/Documents/eddy_tracking/'
    if ctd_type == 'hydro':
        inACEidx_mat = loadmat(dir+'inACEidx_hy.mat')
        inCEidx_mat = loadmat(dir+'inCEidx_hy.mat')
        ctdinACEidx_mat = loadmat(dir+'ctdinACEidx_hy.mat')
        ctdinCEidx_mat = loadmat(dir+'ctdinCEidx_hy.mat')

        inACEidx = inACEidx_mat['inACEidx_hy']-1   # due to matlab to python index
        inCEidx = inCEidx_mat['inCEidx_hy']-1      # due to matlab to python index
        ctdinACEidx = ctdinACEidx_mat['ctdinACEidx_hy']
        ctdinCEidx = ctdinCEidx_mat['ctdinCEidx_hy']
    elif ctd_type == 'ctd':
        inACEidx_mat = loadmat(dir+'inACEidx.mat')
        inCEidx_mat = loadmat(dir+'inCEidx.mat')
        ctdinACEidx_mat = loadmat(dir+'ctdinACEidx.mat')
        ctdinCEidx_mat = loadmat(dir+'ctdinCEidx.mat')

        inACEidx = inACEidx_mat['inACEidx']-1   # due to matlab to python index
        inCEidx = inCEidx_mat['inCEidx']-1      # due to matlab to python index
        ctdinACEidx = ctdinACEidx_mat['ctdinACEidx']
        ctdinCEidx = ctdinCEidx_mat['ctdinCEidx']
    else:
        print('invalid input')

    if eddy_type == 'C':
        profidx = ctdinCEidx
        eddyidx = inCEidx
    elif eddy_type == 'A':
        profidx = ctdinACEidx
        eddyidx = inACEidx
    
    return profidx,eddyidx

def importHisProf(ctd_type):
    # also include renameing n_prof, and calculating sigma0
    # import prof
    if ctd_type == 'ctd':
        prof = xr.open_dataset('/Users/renjiongqiu/Library/CloudStorage/OneDrive-UniversityofTasmania/Documents/Python/extract_data/synchronized/prof_ctd_514prof.nc')
        # rename fluorometer to fluorescence
        prof = prof.rename({'fluorometer':'fluorescence',
                            'fluorometerFlag':'fluorescenceFlag'})
    elif ctd_type == 'hydro':
        prof = xr.open_dataset('/Users/renjiongqiu/Library/CloudStorage/OneDrive-UniversityofTasmania/Documents/Python/extract_data/synchronized/prof_hydro_362prof.nc')
        prof['nitrate'] = prof['nox'] - prof['nitrite']
    else:
        print('invalid input')

    # key_list = list(prof.data_vars.keys())
    # var_list = [il for il in key_list if 'Flag' not in il]
    # for ip in prof['n_prof'].values:
    # for var in var_list:
    #     da = prof[var]
    #     res = 1
    #     pres0 = int(prof.pressure.min())
    #     pres1 = int(prof.pressure.max()+1)
    #     pres_interp = np.arange(pres0,pres1+res/2,res)
    #     value_interp = inpaintnan1d(da,pres0,pres1,res)
    # prof = prof.interp(pressure=pres_interp)
    prof = prof.transpose('pressure','n_prof')
    prof = prof.sortby('time').drop_vars('n_prof').assign_coords({'n_prof':(['n_prof'],np.arange(0,len(prof.ctdindex.values)))})
    # if prof_type == 'ctd':
    #     prof = addSigma0(prof)
    return prof
    
def makeHisProf_type(ctd_type,eddy_type,test=False):
    # prof_type: hydro, ctd
    # eddy_type: CE, ACE

    # import prof
    prof = importHisProf(ctd_type)

    # import eddy
    if eddy_type == 'C':
        eddy = xr.open_dataset('/Users/renjiongqiu/Library/CloudStorage/OneDrive-UniversityofTasmania/Documents/eddy_tracking/cyclonic_eddy_traj_eac_days_more30_meta3p2_dt_2sat.nc', decode_cf=False)
    elif eddy_type == 'A':
        eddy = xr.open_dataset('/Users/renjiongqiu/Library/CloudStorage/OneDrive-UniversityofTasmania/Documents/eddy_tracking/anticyclonic_eddy_traj_eac_days_more30_meta3p2_dt_2sat.nc', decode_cf=False)
    else:
        print('invalid input')
    eddy['effective_radius'].values=eddy['effective_radius'].values.astype('float')

    # import indices
    profidx,eddyidx = importIndices(ctd_type,eddy_type)
    if test:
        l = 5
    else:
        l = len(profidx)

    for ip in range(l):
        # subset
        ictdindex = profidx.flatten()[ip]
        mask_ictd = np.in1d(prof.ctdindex.values,ictdindex)
        prof1 = prof.sel(n_prof=mask_ictd).squeeze()

        ied = eddyidx.flatten()[ip]
        eddy1 = eddy.isel(obs=ied)
        # extract prof parameters
        time = prof1.time.values
        cast = prof1.cast.values
        cruise = prof1.cruise.values
        ctdindex = prof1.ctdindex.values
        lon = prof1.lon.values
        lat = prof1.lat.values

        # extract eddy parameters
        
        lon_edd = eddy1.longitude.values
        lat_edd = eddy1.latitude.values
        lon_edd_eff = eddy1.longitude_max.values
        lat_edd_eff = eddy1.latitude_max.values
        rad = eddy1.effective_radius.values * 50
        rad_deg = rad / 1000 * 0.009
        lon_cont = eddy1.effective_contour_longitude.values.reshape(1,20)*0.01+180
        lat_cont = eddy1.effective_contour_latitude.values.reshape(1,20)*0.01
        time_edd = days2dt64(eddy1.time.values).values
        coords_edd = np.array([lat_edd,lon_edd])
        coords_ctd = np.array([lat,lon])

        # calculate x
        idist = distance.distance(coords_edd, coords_ctd).m
        x = idist/rad

        # interpolate
        key_list = list(prof1.data_vars.keys())
        var_list = [il for il in key_list if 'Flag' not in il]
        pres_old = prof1.pressure.values
        ds = xr.Dataset({})
        for var in var_list:
            res = 1
            var_array = prof1[var].values
            value_interp,pres_interp,obs_n = inpaintnan1d(var_array,pres_old,res)
            value_interp = value_interp.reshape(len(value_interp),1)
            obs_n = obs_n.reshape(len(obs_n),1)
            ds[var] = xr.DataArray(data = value_interp,
                              dims = ['pressure','n_prof'],
                              coords = {'n_prof':(['n_prof'],[ip]),
                                        'cast':(['n_prof'],[cast]),
                                        'cruise':(['n_prof'],[cruise]),
                                        'ctdindex':(['n_prof'],[ctdindex]),
                                        'x':(['n_prof'],[x]),
                                        'pressure':(['pressure'],pres_interp),
                                        'eddyidx':(['n_prof'],[ied]),
                                        'lon':(['n_prof'],[lon]),
                                        'lat':(['n_prof'],[lat]),
                                        'time':(['n_prof'],[time]),
                                        'time_edd':(['n_prof'],time_edd),
                                        'lon_edd':(['n_prof'],[lon_edd]),
                                        'lat_edd':(['n_prof'],[lat_edd]),
                                        'lon_edd_eff':(['n_prof'],[lon_edd_eff]),
                                        'lat_edd_eff':(['n_prof'],[lat_edd_eff]),
                                        'rad':(['n_prof'],[rad]),
                                        'rad_deg':(['n_prof'],[rad_deg])})
            ds[var+'_obs'] = xr.DataArray(data = obs_n,
                              dims = ['pressure','n_prof'],
                              coords = {'n_prof':(['n_prof'],[ip]),
                                        'cast':(['n_prof'],[cast]),
                                        'cruise':(['n_prof'],[cruise]),
                                        'ctdindex':(['n_prof'],[ctdindex]),
                                        'x':(['n_prof'],[x]),
                                        'pressure':(['pressure'],pres_interp),
                                        'eddyidx':(['n_prof'],[ied]),
                                        'lon':(['n_prof'],[lon]),
                                        'lat':(['n_prof'],[lat]),
                                        'time':(['n_prof'],[time]),
                                        'time_edd':(['n_prof'],time_edd),
                                        'lon_edd':(['n_prof'],[lon_edd]),
                                        'lat_edd':(['n_prof'],[lat_edd]),
                                        'lon_edd_eff':(['n_prof'],[lon_edd_eff]),
                                        'lat_edd_eff':(['n_prof'],[lat_edd_eff]),
                                        'rad':(['n_prof'],[rad]),
                                        'rad_deg':(['n_prof'],[rad_deg])})
        ds = ds.assign_coords({'lon_cont':(['n_prof','Ncont'],lon_cont.data),'lat_cont':(['n_prof','Ncont'],lat_cont.data)})
        
        if ip == 0:
            DS = ds.copy()
            # create eddy dataset, use the origin eddy not eddy1
            ds_ed = ieddy_ds(ied,eddy)
            DS_ED = ds_ed.copy()
        else:
            DS = xr.merge([DS,ds])
            # DS = xr.concat([DS,ds],dim='n_prof')

            track = eddy.isel(obs=ied).track.values
            if track not in DS_ED.track.values:
                ds_ed = ieddy_ds(ied,eddy)
                DS_ED = combineEDDY(DS_ED,ds_ed)
    DS = DS.sortby('x')
    # transpose
    DS = DS.transpose('pressure','n_prof','Ncont')
    DS_ED = DS_ED.transpose('n_eddy','obs')
    # convert index to int
    DS['eddyidx'] = DS['eddyidx'].astype(int)
    mask = np.isnan(DS_ED.eddyidx_obs.values)
    DS_ED['eddyidx_obs'] = (['n_eddy','obs'],np.where(mask,-999,DS_ED.eddyidx_obs.values.astype(int)))

    return DS,DS_ED

def CTDaddHydro_his_old(prof_ctd,prof_hydro):

    # check if there is identical 'x' value, proceed if there is not
    if not np.in1d(prof_ctd['x'].values,prof_hydro['x'].values).any():
        ds = xr.merge([prof_ctd.swap_dims({'n_prof':'x'}),prof_hydro.swap_dims({'n_prof':'x'})])
    else:
        print('there is identical x dim')
    
    # drop the old 'n_prof'
    ds = ds.drop_vars('n_prof')

    # create a new 'n_prof' dim and swap with x
    n_prof = np.arange(0,len(ds['x'].values))
    ds = ds.sortby('x') # doesn't have to do this
    ds = ds.assign_coords({'n_prof':(['x'],n_prof)}).swap_dims({'x':'n_prof'})

    return ds

def CTDaddHydro_his(prof_ctd,prof_hydro):

    # create new dim values
    l1 = len(prof_ctd.n_prof.values)
    l2 = len(prof_hydro.n_prof.values)
    n_prof_ar1 = np.arange(0,l1)
    n_prof_ar2 = np.arange(l1,l1+l2)

    # first assign the new dim values to a new coords
    prof_ctd = prof_ctd.assign_coords({'n_prof_new':(['n_prof'],n_prof_ar1)})
    prof_hydro = prof_hydro.assign_coords({'n_prof_new':(['n_prof'],n_prof_ar2)})

    # then swap with the old dim
    prof_ctd = prof_ctd.swap_dims({'n_prof':'n_prof_new'})
    prof_hydro = prof_hydro.swap_dims({'n_prof':'n_prof_new'})
    
    # merge the two profs
    prof_merge = xr.merge([prof_ctd,prof_hydro])
    
    # drop the old dim 'n_prof'
    prof_merge = prof_merge.drop_vars('n_prof')
    prof_merge = prof_merge.rename({'n_prof_new':'n_prof'})

    return prof_merge

def makeHisProf(eddy_type,test=False):
    prof_ctd,eddy_ctd = makeHisProf_type('ctd',eddy_type,test=test)
    prof_hydro,eddy_hydro = makeHisProf_type('hydro',eddy_type,test=test)
    prof_his = CTDaddHydro_his(prof_ctd,prof_hydro)
    prof_his = prof_his.assign({'cast':(['n_prof'],prof_his.cast.values.astype(int)),
                                'eddyidx':(['n_prof'],prof_his.eddyidx.values.astype(int))})
    eddy_his = combineEDDY(eddy_ctd,eddy_hydro)
    return prof_his,eddy_his

def makeHisProf_OUT(eddy_type='O',test=False):
    prof = {}
    for ctd_type in ['ctd','hydro']:
        prof_all = importHisProf(ctd_type)
        profidx_c, _ = importIndices(ctd_type,'C')
        profidx_a, _ = importIndices(ctd_type,'A')
        mask_c = np.in1d(prof_all.ctdindex.values,profidx_c.flatten())
        mask_a = np.in1d(prof_all.ctdindex.values,profidx_a.flatten())
        mask_o = ~( mask_c | mask_a )
        prof_out = prof_all.sel({'n_prof':mask_o}).transpose('pressure','n_prof')

        if test:
            l = 5
        else:
            l = len(prof_out.n_prof.values)

        for i in range(l):
            prof1 = prof_out.isel(n_prof=i).squeeze()
            # extract prof parameters
            n_prof = prof1.n_prof.values
            time = prof1.time.values
            cast = prof1.cast.values
            cruise = prof1.cruise.values
            ctdindex = prof1.ctdindex.values
            lon = prof1.lon.values
            lat = prof1.lat.values

            key_list = list(prof1.data_vars.keys())
            var_list = [il for il in key_list if 'Flag' not in il]
            pres_old = prof1.pressure.values
            ds = xr.Dataset({})
            for var in var_list:
                res = 1
                var_array = prof1[var].values
                value_interp,pres_interp,obs_n = inpaintnan1d(var_array,pres_old,res)
                value_interp = value_interp.reshape(len(value_interp),1)
                obs_n = obs_n.reshape(len(obs_n),1)
                ds[var] = xr.DataArray(data = value_interp,
                                dims = ['pressure','n_prof'],
                                coords = {'n_prof':(['n_prof'],[n_prof]),
                                          'cast':(['n_prof'],[cast]),
                                          'cruise':(['n_prof'],[cruise]),
                                          'ctdindex':(['n_prof'],[ctdindex]),
                                          'pressure':(['pressure'],pres_interp),
                                          'lon':(['n_prof'],[lon]),
                                          'lat':(['n_prof'],[lat]),
                                          'time':(['n_prof'],[time])})
                ds[var] = xr.DataArray(data = obs_n,
                                dims = ['pressure','n_prof'],
                                coords = {'n_prof':(['n_prof'],[n_prof]),
                                          'cast':(['n_prof'],[cast]),
                                          'cruise':(['n_prof'],[cruise]),
                                          'ctdindex':(['n_prof'],[ctdindex]),
                                          'pressure':(['pressure'],pres_interp),
                                          'lon':(['n_prof'],[lon]),
                                          'lat':(['n_prof'],[lat]),
                                          'time':(['n_prof'],[time])})
            if i == 0:
                DS = ds.copy()
            else:
                DS = xr.merge([DS,ds])
                # DS = xr.concat([DS,ds],dim='n_prof')
        # transpose
        DS = DS.transpose('pressure','n_prof')

        prof[ctd_type] = DS
    
    prof_merge = CTDaddHydro_his(prof['ctd'],prof['hydro'])
    prof_merge = prof_merge.assign({'cast':(['n_prof'],prof_merge.cast.values.astype(int))})
    eddy = xr.Dataset({})

    return prof_merge,eddy

# ----------------------------------- V06 -----------------------------------
def inpolygon(xq, yq, xv, yv):
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    xv = xv.reshape(-1)
    yv = yv.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
    return p.contains_points(q).reshape(shape)

def create_subset(ds, lon_min, lon_max, lat_min, lat_max, lifespan):
    # demarkation of the study region
    lon_min, lon_max, lat_min, lat_max = lon_min, lon_max, lat_min, lat_max
    # Creating a mask of the study region
    subset = ds.sel(obs=(ds.longitude > lon_min) & (ds.longitude < lon_max) 
                    & (ds.latitude > lat_min) & (ds.latitude < lat_max))
    # Creating a subset using the mask 
    subset = ds.isel(obs=np.in1d(ds.track, subset.track))
    # Further applying lifespan filter to eddies 
    subset_life = subset.sel(obs=subset.observation_number > lifespan)
    # saving the final output for further analyses
    # Create the final subset
    subset_final = subset.isel(obs=np.in1d(subset.track, subset_life.track))
    return subset_final

def make_interp_dataset_ctd(ctd_files):
    keys = ['pressure','temperature','salinity','oxygen','par','cdom','transmissometer','fluorometer','obs']
    var_list = keys.copy()
    var_list.remove('pressure')
    flags = [var+'Flag' for var in var_list]
    
    for i,fn in enumerate(ctd_files):
        var_dict = {}
        ds = xr.open_dataset(fn)
        lon = ds.longitude.values[0]
        lat = ds.latitude.values[0]
        time = ds.time.values[0]
        ds = ds[keys+flags].squeeze(dim=['time','latitude','longitude'])
        pressure_old = ds.pressure.values
        
        for iv,var in enumerate(var_list):
            da = ds[var]
            daflag = ds[var+'Flag']
            da = da.where(daflag==0,np.nan) # flag the bad data
            id = int(ds.attrs['Deployment'])
            res = 1
            value_interp,pressure_interp,obs_n = inpaintnan1d(da.values,pressure_old,res)
            var_dict[var] = {'data':value_interp,
                             'dims':'pressure',
                             'coords':{'pressure':{'dims':'pressure','data':pressure_interp}}}
            if var in ['temperature','salinity','oxygen','fluorometer']:
                var_dict[var+'_obs'] = {'data':obs_n,
                                'dims':'pressure',
                                'coords':{'pressure':{'dims':'pressure','data':pressure_interp}}}
        
        ds = xr.Dataset.from_dict(var_dict)
        ds = ds.expand_dims({'deployment':1})
        ds['deployment'] = [id]
        ds = ds.assign_coords( {'lat':(['deployment'],[lat]),
                                'lon':(['deployment'],[lon]),
                                'time':(['deployment'],[time]),
                                'pressure':(['pressure'],pressure_interp)} )
        ds = ds.drop_duplicates(dim=...)
        if i == 0:
            DS = ds.copy()
        else:
            DS = xr.merge([DS,ds])
    DS = DS.transpose('pressure','deployment')

    return DS

def make_interp_dataset_hydro(hydro_file):
    df = pd.read_csv(hydro_file)
    # df['Nitrate (uM)'] = df['NOx (uM)'] - df['Nitrite (uM)']
    df.rename(columns={'Deployment':'deployment',
                    'Time':'time',
                    'Latitude':'lat',
                    'Longitude':'lon',
                    'Pressure':'pressure',
                    'Temperature (degC)':'temperature',
                    'Temperature flag':'temperatureFlag',
                    'Salinity (PSU)':'salinity',
                    'Salinity flag':'salinityFlag',
                    'Oxygen (uM)':'oxygen',
                    'Oxygen flag':'oxygenFlag',
                    'NOx (uM)':'nox',
                    'NOx flag':'noxFlag',
                    'Phosphate (uM)':'phosphate',
                    'Phosphate flag':'phosphateFlag',
                    'Silicate (uM)':'silicate',
                    'Silicate flag':'silicateFlag',
                    'Ammonia (uM)':'ammonia',
                    'Ammonia flag':'ammoniaFlag',
                    'Nitrite (uM)':'nitrite',
                    'Nitrite flag':'nitriteFlag'},inplace=True)
    df['nitrate'] = df['nox'] - df['nitrite']
    ctd_idx = np.unique(df.deployment.values)
    keys = ['pressure','temperature','salinity','oxygen','nox','nitrite','nitrate','ammonia','phosphate','silicate']
    var_list = keys.copy()
    var_list.remove('pressure')
    
    for i,id in enumerate(ctd_idx):
        var_dict = {}
        lon = np.mean(df.loc[df.deployment==id,'lon'])
        lat = np.mean(df.loc[df.deployment==id,'lat'])
        time = df.loc[df.deployment==id,'time'].values.astype('datetime64[ns]').astype('float64').mean().astype('datetime64[ns]')
        df_i = df.loc[df.deployment==id,keys]
        df_i.set_index('pressure',inplace=True)
        ds_i = df_i.to_xarray()
        for iv,var in enumerate(var_list):
            da = ds_i[var]
            res = 1
            pressure_old = da.pressure.values
            value_interp,pressure_interp,obs_n = inpaintnan1d(da.values,pressure_old,res)
            var_dict[var] = {'data':value_interp,
                             'dims':'pressure',
                             'coords':{'pressure':{'dims':'pressure','data':pressure_interp}}}
            if var in ['nitrate','phosphate','silicate']:
                var_dict[var+'_obs'] = {'data':obs_n,
                                'dims':'pressure',
                                'coords':{'pressure':{'dims':'pressure','data':pressure_interp}}}
        ds = xr.Dataset.from_dict(var_dict)
        ds = ds.expand_dims({'deployment':1})
        ds['deployment'] = [id]
        ds = ds.assign_coords( {'lat':(['deployment'],[lat]),
                                'lon':(['deployment'],[lon]),
                                'time':(['deployment'],[time]),
                                'pressure':(['pressure'],pressure_interp)} )
        ds = ds.drop_duplicates(dim=...)
        if i == 0:
            DS = ds.copy()
        else:
            DS = xr.merge([DS,ds])
    DS = DS.transpose('pressure','deployment')

    return DS

# align the coords of ctd and hydro
def CTDaddHydro(prof_ctd,prof_hydro):
    deployment_intersect = list(set(prof_ctd.deployment.values) & set(prof_hydro.deployment.values))
    coords_sel = list(set(prof_ctd.coords.keys()) & set(prof_hydro.coords.keys()))
    coords_sel = [ico for ico in coords_sel if 'deployment' in prof_ctd[ico].dims and 'deployment' in prof_hydro[ico].dims ]
    for d in deployment_intersect:
        for ico in coords_sel:
            if not ico == 'time':
                coords_mean = np.nanmean( [prof_ctd.sel({'deployment':prof_ctd.deployment==d})[ico].values[0],prof_hydro.sel({'deployment':prof_hydro.deployment==d})[ico].values[0]] )
            elif ico == 'time':
                coords_mean = np.datetime64( pd.Series([prof_ctd.sel({'deployment':prof_ctd.deployment==d})[ico].values[0],prof_hydro.sel({'deployment':prof_hydro.deployment==d})[ico].values[0]]).mean() )
            else:
                print(f'no matching coords to {ico}')
            prof_ctd.sel({'deployment':prof_ctd.deployment==d}).assign_coords({ico:(['deployment'],[coords_mean])})
            prof_hydro.sel({'deployment':prof_hydro.deployment==d}).assign_coords({ico:(['deployment'],[coords_mean])})
    prof_new = xr.merge([prof_ctd,prof_hydro],compat='override') # keep temp,salinity,oxygen,sigma0 from ctd 
    return prof_new

def importEddyV06(eddy_type):
    if eddy_type == 'C':
        track = 131686
        ds = xr.open_dataset('/Users/renjiongqiu/Library/CloudStorage/OneDrive-UniversityofTasmania/Documents/eddy_tracking/Eddy_trajectory_nrt_3.2exp_cyclonic_20180101_20231109.nc', decode_cf=False)
    elif eddy_type =='A':
        track = 141247
        ds = xr.open_dataset('/Users/renjiongqiu/Library/CloudStorage/OneDrive-UniversityofTasmania/Documents/eddy_tracking/Eddy_trajectory_nrt_3.2exp_anticyclonic_20180101_20231109.nc', decode_cf=False)
    subset = create_subset(ds, 150, 159, -40, -25, 30)
    eddy = subset.sel(obs=(subset.track==track))
    eddyidx_obs = np.arange(0,len(eddy.amplitude.values))
    eddy = eddy.assign_coords({'eddyidx_obs':(['obs'],eddyidx_obs)})
    return eddy

def profineddy(prof,eddy):
    eddy['effective_radius'].values=eddy['effective_radius'].values.astype('float')
    days = eddy.time.values*1.1574074074074073e-05*timedelta(days=1)
    dt = datetime(1950,1,1)+days
    dt64 = pd.Series(dt,dtype='datetime64[ns]')
    t_ctd = pd.Series(prof.time,dtype='datetime64[ns]').dt.round('D')
    ix = 0
    for it,t in enumerate(t_ctd):
        idx = abs(dt64-t).argmin()
        et = dt64[idx]
        ilon_edd = eddy.sel(obs=(dt64==et)).longitude.values
        ilat_edd = eddy.sel(obs=(dt64==et)).latitude.values
        ilon_edd_eff = eddy.sel(obs=(dt64==et)).longitude_max.values
        ilat_edd_eff = eddy.sel(obs=(dt64==et)).latitude_max.values
        ilon_cont = eddy.sel(obs=(dt64==et)).effective_contour_longitude.values*0.01+180
        ilat_cont = eddy.sel(obs=(dt64==et)).effective_contour_latitude.values*0.01
        ilon_ctd = prof.lon[it].values
        ilat_ctd = prof.lat[it].values
        coords_edd = (ilat_edd,ilon_edd)
        coords_ctd = (ilat_ctd,ilon_ctd)
        if inpolygon(ilon_ctd,ilat_ctd,ilon_cont,ilat_cont):
            ied = eddy.sel(obs=(dt64==et))['eddyidx_obs'].values
            rad = eddy.sel(obs=(dt64==et)).effective_radius.values * 50
            rad_deg = rad / 1000 * 0.009
            dist = distance.distance(coords_edd, coords_ctd).m
            x = dist/rad
            ds1 = prof.isel(deployment=[it])
            ds1 = ds1.assign_coords( {'lat':(['deployment'],[ilat_ctd]),
                                        'lon':(['deployment'],[ilon_ctd]),
                                        'eddyidx':(['deployment'],ied),
                                        'x':(['deployment'],x),
                                        'time':(['deployment'],[prof.time[it].values]),
                                        'time_edd':(['deployment'],[et]),
                                        'lon_edd':(['deployment'],ilon_edd),
                                        'lat_edd':(['deployment'],ilat_edd),
                                        'lon_edd_eff':(['deployment'],ilon_edd_eff),
                                        'lat_edd_eff':(['deployment'],ilat_edd_eff),
                                        'rad':(['deployment'],rad),
                                        'rad_deg':(['deployment'],rad_deg),
                                        'lon_cont':(['deployment','Ncont'],ilon_cont.reshape(1,20)),
                                        'lat_cont':(['deployment','Ncont'],ilat_cont.reshape(1,20))} )
            ds1 = ds1.drop_duplicates(dim=...)
            
            if ix == 0:
                DS = ds1.copy()
            else:
                DS = xr.merge([DS,ds1])
            ix += 1
    # make deployment as a coords and swap n_prof with deployment
    DS = DS.assign_coords({'n_prof':(['deployment'],np.arange(0,len(DS['deployment'].values)))}).swap_dims({'deployment':'n_prof'})
    DS = DS.sortby('n_prof')
    DS = DS.transpose('pressure','n_prof','Ncont')
    DS['n_prof'] = DS['n_prof'].astype(int)
    return DS

def makeV06Prof(eddy_type):

    # import and interpolate sensor data
    path2data = '/Users/renjiongqiu/Library/CloudStorage/OneDrive-UniversityofTasmania/Documents/aa_in2023_v06/data_ship/ctd_data/processing/IN2023_V06/cap/cappro/avg'
    ctd_files = glob(os.path.join(path2data, '*.nc'))
    prof_ctd = make_interp_dataset_ctd(ctd_files)
    
    # import and interpolate hydro data
    hydro_file = '/Users/renjiongqiu/Library/CloudStorage/OneDrive-UniversityofTasmania/Documents/aa_in2023_v06/data_ship/Hydrochem/PRELIM_in2023_v06HydroCTD.csv'
    prof_hydro = make_interp_dataset_hydro(hydro_file)

    # combine ctd and hydro
    prof_combined = CTDaddHydro(prof_ctd,prof_hydro)
    # prof_combined = prof_combined.assign({'eddyidx':(['n_prof'],prof_combined.eddyidx.values.astype(int))})

    # import the two eddies in V06 cruise
    eddy = importEddyV06(eddy_type)
    eddy_ds = ieddy_ds_v06(eddy)

    # collocate the profiles and eddies
    prof_in_eddy = profineddy(prof_combined,eddy)

    # change fluorometer to fluorescence
    prof_in_eddy = prof_in_eddy.rename({'fluorometer':'fluorescence',
                                        'fluorometer_obs':'fluorescence_obs'})

    return prof_in_eddy,eddy_ds

def makeV06Prof_OUT(eddy_type='O'):

    # import and interpolate sensor data
    path2data = '/Users/renjiongqiu/Library/CloudStorage/OneDrive-UniversityofTasmania/Documents/aa_in2023_v06/data_ship/ctd_data/processing/IN2023_V06/cap/cappro/avg'
    ctd_files = glob(os.path.join(path2data, '*.nc'))
    prof_ctd = make_interp_dataset_ctd(ctd_files)
    
    # import and interpolate hydro data
    hydro_file = '/Users/renjiongqiu/Library/CloudStorage/OneDrive-UniversityofTasmania/Documents/aa_in2023_v06/data_ship/Hydrochem/PRELIM_in2023_v06HydroCTD.csv'
    prof_hydro = make_interp_dataset_hydro(hydro_file)

    # combine ctd and hydro
    prof_combined = CTDaddHydro(prof_ctd,prof_hydro)

    # import the two eddies in V06 cruise
    eddy_ce = importEddyV06('C')
    eddy_ae = importEddyV06('A')

    # collocate the profiles and eddies
    prof_in_ce = profineddy(prof_combined,eddy_ce)
    prof_in_ae = profineddy(prof_combined,eddy_ae)

    # select the prof outside of eddies
    mask_c = np.in1d(prof_combined.deployment.values,prof_in_ce.deployment.values)
    mask_a = np.in1d(prof_combined.deployment.values,prof_in_ae.deployment.values)
    mask_o = ~( mask_c | mask_a )
    prof_out_eddy = prof_combined.sel({'deployment':mask_o})
    prof_out_eddy = prof_out_eddy.assign_coords({'n_prof':(['deployment'],np.arange(0,len(prof_out_eddy['deployment'].values)))}).swap_dims({'deployment':'n_prof'})
    prof_out_eddy = prof_out_eddy.sortby('n_prof')
    prof_out_eddy = prof_out_eddy.transpose('pressure','n_prof')
    prof_out_eddy['n_prof'] = prof_out_eddy['n_prof'].astype(int)

    # change fluorometer to fluorescence
    prof_out_eddy = prof_out_eddy.rename({'fluorometer':'fluorescence',
                                          'fluorometer_obs':'fluorescence_obs'})

    eddy_ds = xr.Dataset({})
    return prof_out_eddy,eddy_ds
    
def ieddy_ds_v06(eddy):
    track = eddy.isel(obs=0).track.values
    keys = ['amplitude','time','effective_area','latitude','longitude']
    eddy1 = eddy[keys]
    # coords
    eddyidx_obs = eddy1['eddyidx_obs'].values
    l = len(eddyidx_obs)
    # data_vars, 2D
    amplitude = eddy1['amplitude'].values * 0.0001
    area = eddy1['effective_area'].values
    time = days2dt64(eddy1['time'].values)
    age = time - min(time)
    lon = eddy1.longitude.values
    lat = eddy1.latitude.values
    # store time datetime64[ns] into int for padding
    time = np.array(time).astype(int)
    age = np.array(age.dt.days)
    # data_vars, 1D
    amplitude_t = np.round(max(amplitude),4)
    area_t = max(area)
    time_birth = min(time)
    lat_birth = min(lat)
    lon_birth = max(lon)
    longevity = max(age)

    ds = xr.Dataset(
        data_vars = {
                    #  2D vars
                    'amplitude': ( ['n_eddy','obs'],amplitude.reshape(1,l) ),
                    'area':      ( ['n_eddy','obs'],area.reshape(1,l) ),
                    'time':      ( ['n_eddy','obs'],np.array(time).reshape(1,l) ),
                    'age':       ( ['n_eddy','obs'],np.array(age).reshape(1,l) ),
                    'lon':       ( ['n_eddy','obs'],lon.reshape(1,l) ),
                    'lat':       ( ['n_eddy','obs'],lat.reshape(1,l) ),
                    #  1D vars
                    'amplitude_t':( ['n_eddy'],[amplitude_t] ),
                    'area_t':     ( ['n_eddy'],[area_t] ),
                    'time_birth': ( ['n_eddy'],[time_birth] ),
                    'longevity':  ( ['n_eddy'],[longevity] ),
                    'lat_birth':  ( ['n_eddy'],[lat_birth] ),
                    'lon_birth':  ( ['n_eddy'],[lon_birth] ),
                    'track':      ( ['n_eddy'],[track] )},
        
        coords = {'eddyidx_obs' : ( ['n_eddy','obs'],eddyidx_obs.reshape(1,l) )}
    )
    return ds