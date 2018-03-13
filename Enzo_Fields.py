import yt
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.table import Table , Column ,vstack,hstack
from scipy.ndimage.filters import gaussian_filter as difussion
import os
import glob
from yt.utilities.physical_constants import G
from yt.units import kpc,pc,km,second,yr,Myr

from yt.fields.derived_field import \
    ValidateGridType, \
    ValidateParameter, \
    ValidateSpatial, \
    NeedsParameter
import h5py
from yt.funcs import \
    just_one
from common_functions import *

sl_left = slice(None, -2, None)
sl_right = slice(2, None, None)
div_fac = 2.0

sl_center = slice(1, -1, None)
ftype='gas'

vort_validators = [ValidateSpatial(1,
                        [(ftype, "velocity_x"),
                         (ftype, "velocity_y"),
                         (ftype, "velocity_z")])]



def _Disk_H(field, data):
    center = data.get_field_parameter('center')
    z = data["z"] - center[2]
    return np.abs(z)

def _vturb(field, data):
    fx  = data[ftype, "velocity_x"][sl_right,sl_center,sl_center]/6.0
    fx += data[ftype, "velocity_x"][sl_left,sl_center,sl_center]/6.0
    fx += data[ftype, "velocity_x"][sl_center,sl_right,sl_center]/6.0
    fx += data[ftype, "velocity_x"][sl_center,sl_left,sl_center]/6.0
    fx += data[ftype, "velocity_x"][sl_center,sl_center,sl_right]/6.0
    fx += data[ftype, "velocity_x"][sl_center,sl_center,sl_left]/6.0

    FX= (data[ftype, "velocity_x"][sl_center,sl_center,sl_center]-fx)

    fy  = data[ftype, "velocity_y"][sl_right,sl_center,sl_center]/6.0
    fy += data[ftype, "velocity_y"][sl_left,sl_center,sl_center]/6.0
    fy += data[ftype, "velocity_y"][sl_center,sl_right,sl_center]/6.0
    fy += data[ftype, "velocity_y"][sl_center,sl_left,sl_center]/6.0
    fy += data[ftype, "velocity_y"][sl_center,sl_center,sl_right]/6.0
    fy += data[ftype, "velocity_y"][sl_center,sl_center,sl_left]/6.0

    FY= (data[ftype, "velocity_y"][sl_center,sl_center,sl_center]-fy)

    fz  = data[ftype, "velocity_z"][sl_right,sl_center,sl_center]/6.0
    fz += data[ftype, "velocity_z"][sl_left,sl_center,sl_center]/6.0
    fz += data[ftype, "velocity_z"][sl_center,sl_right,sl_center]/6.0
    fz += data[ftype, "velocity_z"][sl_center,sl_left,sl_center]/6.0
    fz += data[ftype, "velocity_z"][sl_center,sl_center,sl_right]/6.0
    fz += data[ftype, "velocity_z"][sl_center,sl_center,sl_left]/6.0

    FZ= (data[ftype, "velocity_z"][sl_center,sl_center,sl_center]-fz)



    new_field = data.ds.arr(np.zeros_like(data[ftype, "velocity_z"],
                                          dtype=np.float64),
                            FX.units)
    new_field[sl_center, sl_center, sl_center] = np.sqrt(FX**2+FY**2+FZ**2)
    return new_field

def _sturb(field, data):
    fx  = data[ftype, "velocity_x"][sl_right,sl_center,sl_center]/7.0
    fx += data[ftype, "velocity_x"][sl_left,sl_center,sl_center]/7.0
    fx += data[ftype, "velocity_x"][sl_center,sl_right,sl_center]/7.0
    fx += data[ftype, "velocity_x"][sl_center,sl_left,sl_center]/7.0
    fx += data[ftype, "velocity_x"][sl_center,sl_center,sl_right]/7.0
    fx += data[ftype, "velocity_x"][sl_center,sl_center,sl_left]/7.0
    fx += data[ftype, "velocity_x"][sl_center,sl_center,sl_center]/7.0

    FX  = (data[ftype, "velocity_x"][sl_right,sl_center,sl_center]-fx)**2/7.0
    FX += (data[ftype, "velocity_x"][sl_left,sl_center,sl_center]-fx)**2/7.0
    FX += (data[ftype, "velocity_x"][sl_center,sl_right,sl_center]-fx)**2/7.0
    FX += (data[ftype, "velocity_x"][sl_center,sl_left,sl_center]-fx)**2/7.0
    FX += (data[ftype, "velocity_x"][sl_center,sl_center,sl_right]-fx)**2/7.0
    FX += (data[ftype, "velocity_x"][sl_center,sl_center,sl_left]-fx)**2/7.0
    FX += (data[ftype, "velocity_x"][sl_center,sl_center,sl_center]-fx)**2/7.0

    fy  = data[ftype, "velocity_y"][sl_right,sl_center,sl_center]/7.0
    fy += data[ftype, "velocity_y"][sl_left,sl_center,sl_center]/7.0
    fy += data[ftype, "velocity_y"][sl_center,sl_right,sl_center]/7.0
    fy += data[ftype, "velocity_y"][sl_center,sl_left,sl_center]/7.0
    fy += data[ftype, "velocity_y"][sl_center,sl_center,sl_right]/7.0
    fy += data[ftype, "velocity_y"][sl_center,sl_center,sl_left]/7.0
    fy += data[ftype, "velocity_y"][sl_center,sl_center,sl_center]/7.0

    FY  = (data[ftype, "velocity_y"][sl_right,sl_center,sl_center]-fy)**2/7.0
    FY += (data[ftype, "velocity_y"][sl_left,sl_center,sl_center]-fy)**2/7.0
    FY += (data[ftype, "velocity_y"][sl_center,sl_right,sl_center]-fy)**2/7.0
    FY += (data[ftype, "velocity_y"][sl_center,sl_left,sl_center]-fy)**2/7.0
    FY += (data[ftype, "velocity_y"][sl_center,sl_center,sl_right]-fy)**2/7.0
    FY += (data[ftype, "velocity_y"][sl_center,sl_center,sl_left]-fy)**2/7.0
    FY += (data[ftype, "velocity_y"][sl_center,sl_center,sl_center]-fy)**2/7.0

    fz  = data[ftype, "velocity_z"][sl_right,sl_center,sl_center]/7.0
    fz += data[ftype, "velocity_z"][sl_left,sl_center,sl_center]/7.0
    fz += data[ftype, "velocity_z"][sl_center,sl_right,sl_center]/7.0
    fz += data[ftype, "velocity_z"][sl_center,sl_left,sl_center]/7.0
    fz += data[ftype, "velocity_z"][sl_center,sl_center,sl_right]/7.0
    fz += data[ftype, "velocity_z"][sl_center,sl_center,sl_left]/7.0
    fz += data[ftype, "velocity_z"][sl_center,sl_center,sl_center]/7.0

    FZ  = (data[ftype, "velocity_z"][sl_right,sl_center,sl_center]-fz)**2/7.0
    FZ += (data[ftype, "velocity_z"][sl_left,sl_center,sl_center]-fz)**2/7.0
    FZ += (data[ftype, "velocity_z"][sl_center,sl_right,sl_center]-fz)**2/7.0
    FZ += (data[ftype, "velocity_z"][sl_center,sl_left,sl_center]-fz)**2/7.0
    FZ += (data[ftype, "velocity_z"][sl_center,sl_center,sl_right]-fz)**2/7.0
    FZ += (data[ftype, "velocity_z"][sl_center,sl_center,sl_left]-fz)**2/7.0
    FZ += (data[ftype, "velocity_z"][sl_center,sl_center,sl_center]-fz)**2/7.0

    new_field = data.ds.arr(np.zeros_like(data[ftype, "velocity_z"],
                                          dtype=np.float64),
                            np.sqrt(FX).units)
    new_field[sl_center, sl_center, sl_center] = np.sqrt(FX+FY+FZ)
    return new_field

def _wturb(field, data):

    td  = data[ftype, "density"][sl_right,sl_center,sl_center]
    td += data[ftype, "density"][sl_left,sl_center,sl_center]
    td += data[ftype, "density"][sl_center,sl_right,sl_center]
    td += data[ftype, "density"][sl_center,sl_left,sl_center]
    td += data[ftype, "density"][sl_center,sl_center,sl_right]
    td += data[ftype, "density"][sl_center,sl_center,sl_left]
    td += data[ftype, "density"][sl_center,sl_center,sl_center]

    fx  = data[ftype, "px"][sl_right,sl_center,sl_center]/td
    fx += data[ftype, "px"][sl_left,sl_center,sl_center]/td
    fx += data[ftype, "px"][sl_center,sl_right,sl_center]/td
    fx += data[ftype, "px"][sl_center,sl_left,sl_center]/td
    fx += data[ftype, "px"][sl_center,sl_center,sl_right]/td
    fx += data[ftype, "px"][sl_center,sl_center,sl_left]/td
    fx += data[ftype, "px"][sl_center,sl_center,sl_center]/td

    FX  = (data[ftype, "velocity_x"][sl_right,sl_center,sl_center] -fx)**2*data[ftype, "density"][sl_right,sl_center,sl_center]
    FX += (data[ftype, "velocity_x"][sl_left,sl_center,sl_center]  -fx)**2*data[ftype, "density"][sl_left,sl_center,sl_center]
    FX += (data[ftype, "velocity_x"][sl_center,sl_right,sl_center] -fx)**2*data[ftype, "density"][sl_center,sl_right,sl_center]
    FX += (data[ftype, "velocity_x"][sl_center,sl_left,sl_center]  -fx)**2*data[ftype, "density"][sl_center,sl_left,sl_center]
    FX += (data[ftype, "velocity_x"][sl_center,sl_center,sl_right] -fx)**2*data[ftype, "density"][sl_center,sl_center,sl_right]
    FX += (data[ftype, "velocity_x"][sl_center,sl_center,sl_left]  -fx)**2*data[ftype, "density"][sl_center,sl_center,sl_left]
    FX += (data[ftype, "velocity_x"][sl_center,sl_center,sl_center]-fx)**2*data[ftype, "density"][sl_center,sl_center,sl_center]

    fy  = data[ftype, "py"][sl_right,sl_center,sl_center]/td
    fy += data[ftype, "py"][sl_left,sl_center,sl_center]/td
    fy += data[ftype, "py"][sl_center,sl_right,sl_center]/td
    fy += data[ftype, "py"][sl_center,sl_left,sl_center]/td
    fy += data[ftype, "py"][sl_center,sl_center,sl_right]/td
    fy += data[ftype, "py"][sl_center,sl_center,sl_left]/td
    fy += data[ftype, "py"][sl_center,sl_center,sl_center]/td

    FY  = (data[ftype, "velocity_y"][sl_right,sl_center,sl_center] -fy)**2*data[ftype, "density"][sl_right,sl_center,sl_center]
    FY += (data[ftype, "velocity_y"][sl_left,sl_center,sl_center]  -fy)**2*data[ftype, "density"][sl_left,sl_center,sl_center]
    FY += (data[ftype, "velocity_y"][sl_center,sl_right,sl_center] -fy)**2*data[ftype, "density"][sl_center,sl_right,sl_center]
    FY += (data[ftype, "velocity_y"][sl_center,sl_left,sl_center]  -fy)**2*data[ftype, "density"][sl_center,sl_left,sl_center]
    FY += (data[ftype, "velocity_y"][sl_center,sl_center,sl_right] -fy)**2*data[ftype, "density"][sl_center,sl_center,sl_right]
    FY += (data[ftype, "velocity_y"][sl_center,sl_center,sl_left]  -fy)**2*data[ftype, "density"][sl_center,sl_center,sl_left]
    FY += (data[ftype, "velocity_y"][sl_center,sl_center,sl_center]-fy)**2*data[ftype, "density"][sl_center,sl_center,sl_center]

    fz  = data[ftype, "pz"][sl_right,sl_center,sl_center]/td
    fz += data[ftype, "pz"][sl_left,sl_center,sl_center]/td
    fz += data[ftype, "pz"][sl_center,sl_right,sl_center]/td
    fz += data[ftype, "pz"][sl_center,sl_left,sl_center]/td
    fz += data[ftype, "pz"][sl_center,sl_center,sl_right]/td
    fz += data[ftype, "pz"][sl_center,sl_center,sl_left]/td
    fz += data[ftype, "pz"][sl_center,sl_center,sl_center]/td

    FZ  = (data[ftype, "velocity_z"][sl_right,sl_center,sl_center] -fz)**2*data[ftype, "density"][sl_right,sl_center,sl_center]
    FZ += (data[ftype, "velocity_z"][sl_left,sl_center,sl_center]  -fz)**2*data[ftype, "density"][sl_left,sl_center,sl_center]
    FZ += (data[ftype, "velocity_z"][sl_center,sl_right,sl_center] -fz)**2*data[ftype, "density"][sl_center,sl_right,sl_center]
    FZ += (data[ftype, "velocity_z"][sl_center,sl_left,sl_center]  -fz)**2*data[ftype, "density"][sl_center,sl_left,sl_center]
    FZ += (data[ftype, "velocity_z"][sl_center,sl_center,sl_right] -fz)**2*data[ftype, "density"][sl_center,sl_center,sl_right]
    FZ += (data[ftype, "velocity_z"][sl_center,sl_center,sl_left]  -fz)**2*data[ftype, "density"][sl_center,sl_center,sl_left]
    FZ += (data[ftype, "velocity_z"][sl_center,sl_center,sl_center]-fz)**2*data[ftype, "density"][sl_center,sl_center,sl_center]

    FX/=td
    FY/=td
    FZ/=td
    new_field = data.ds.arr(np.zeros_like(data[ftype, "velocity_z"],
                                          dtype=np.float64),
                            np.sqrt(FX).units)
    new_field[sl_center, sl_center, sl_center] = np.sqrt(FX+FY+FZ)
    return new_field

def _vz_squared(field,data):
        if data.has_field_parameter("bulk_velocity"):
                bv = data.get_field_parameter("bulk_velocity").in_units("cm/s")
        else:
                bv = data.ds.arr(np.zeros(3), "cm/s")
        vz = data["gas","velocity_z"] - bv[2]

        return vz**2

def _vz(field,data):
        if data.has_field_parameter("bulk_velocity"):
                bv = data.get_field_parameter("bulk_velocity").in_units("cm/s")
        else:
                bv = data.ds.arr(np.zeros(3), "cm/s")
        vz = data["gas","velocity_z"] - bv[2]

        return vz

def _px(field,data):
        if data.has_field_parameter("bulk_velocity"):
                bv = data.get_field_parameter("bulk_velocity").in_units("cm/s")
        else:
                bv = data.ds.arr(np.zeros(3), "cm/s")
        vx = data["gas","velocity_x"] - bv[0]
        return vx*data["gas","density"]

def _py(field,data):
        if data.has_field_parameter("bulk_velocity"):
                bv = data.get_field_parameter("bulk_velocity").in_units("cm/s")
        else:
                bv = data.ds.arr(np.zeros(3), "cm/s")
        vx = data["gas","velocity_y"] - bv[0]
        return vx*data["gas","density"]

def _pz(field,data):
        if data.has_field_parameter("bulk_velocity"):
                bv = data.get_field_parameter("bulk_velocity").in_units("cm/s")
        else:
                bv = data.ds.arr(np.zeros(3), "cm/s")
        vx = data["gas","velocity_z"] - bv[0]
        return vx*data["gas","density"]

def _sound_speed(field, data):
    tr = data.ds.gamma * data["gas", "pressure"] / data["gas", "density"]
    return np.sqrt(tr)

def _sound_speed_2(field, data):
    tr = data.ds.gamma * data["gas", "pressure"] / data["gas", "density"]
    return tr

def _sound_speed_rep(field, data):
    tr = data.ds.gamma * data["gas", "pressure"] / data["gas", "density"]
    return 1.0/np.sqrt(tr)

def _sound_speed_rep_2(field, data):
    tr = data.ds.gamma * data["gas", "pressure"] / data["gas", "density"]
    return 1.0/tr

yt.add_field("Disk_H",
             function=_Disk_H,
             units="pc",
             take_log=False,
             validators=[ValidateParameter('center')])
yt.add_field(("gas", "vz_squared"),function=_vz_squared,units="km**2/s**2")
yt.add_field(("gas", "vz"),function=_vz,units="km/s")
yt.add_field("sound_speed", function=_sound_speed, units=r"km/s")
yt.add_field("sound_speed_rep", function=_sound_speed_rep, units=r"s/km")
yt.add_field("sound_speed_2", function=_sound_speed_2, units=r"km**2/s**2")
yt.add_field("sound_speed_rep_2", function=_sound_speed_rep_2, units=r"s**2/km**2")
yt.add_field(("gas", "px"),function=_px,units="g/s/cm**2")
yt.add_field(("gas", "py"),function=_py,units="g/s/cm**2")
yt.add_field(("gas", "pz"),function=_pz,units="g/s/cm**2")
yt.add_field(("gas", "sturb"),function=_sturb,units="km/s",validators=vort_validators)
yt.add_field(("gas", "vturb"),function=_vturb,units="km/s",validators=vort_validators)
yt.add_field(("gas", "wturb"),function=_vturb,units="km/s",validators=vort_validators)

def Turbulence_1D(name):
    if os.path.isfile(name+'_turb_1D.npy'):
        return True
    data='Sims/'+name+'/G-'+name[-4:]
    ds=yt.load(data)

    L=40*kpc

    grids=ds.refine_by**ds.index.max_level*ds.domain_dimensions[0]

    DX=ds.arr(1, 'code_length')
    DX.convert_to_units('pc')
    dr=DX/grids

    dd=ds.all_data()
    print('### Projection ###')
    disk_dd = dd.cut_region(["obj['Disk_H'].in_units('pc') < 1.0e3"])
    proj = ds.proj('vz', 2,data_source=disk_dd,weight_field='density')


    width = (float(L), 'kpc')
    NN=int(L/dr)
    dA=((L/NN)**2).in_units('pc**2')

    res = [NN, NN]
    frb = proj.to_frb(width, res, center=[0.5,0.5,0.5])

    vz=frb['vz'].in_units('km/s')
    vz_squared=frb['vz_squared'].in_units('km**2/s**2')

    sigma_vz=vz_squared-vz**2
    sigma_vz=np.sqrt(np.abs(sigma_vz))
    sigma_vz.convert_to_units('km/s')

    np.save(name+'_turb_1D',sigma_vz)

def Turbulence_3D(name):
    if os.path.isfile(name+'_turb_3D.npy'):
        return True
    data='Sims/'+name+'/G-'+name[-4:]
    ds=yt.load(data)

    L=40*kpc

    grids=ds.refine_by**ds.index.max_level*ds.domain_dimensions[0]

    DX=ds.arr(1, 'code_length')
    DX.convert_to_units('pc')
    dr=DX/grids

    dd=ds.all_data()
    print('### Projection ###')
    disk_dd = dd.cut_region(["obj['Disk_H'].in_units('pc') < 1.0e3"])
    proj = ds.proj('vturb', 2,data_source=disk_dd,weight_field='density')


    width = (float(L), 'kpc')
    NN=int(L/dr)
    dA=((L/NN)**2).in_units('pc**2')

    res = [NN, NN]
    frb = proj.to_frb(width, res, center=[0.5,0.5,0.5])

    print('### Turbulence Map ###')
    v3D=frb['vturb'].in_units('km/s')

    np.save(name+'_turb_3D',v3D)
    return True

def Sigma_3D(name):
    if os.path.isfile(name+'_sigma_3D.npy'):
        return True
    data='Sims/'+name+'/G-'+name[-4:]
    ds=yt.load(data)

    L=40*kpc

    grids=ds.refine_by**ds.index.max_level*ds.domain_dimensions[0]

    DX=ds.arr(1, 'code_length')
    DX.convert_to_units('pc')
    dr=DX/grids

    dd=ds.all_data()
    print('### Projection ###')
    disk_dd = dd.cut_region(["obj['Disk_H'].in_units('pc') < 1.0e3"])
    proj = ds.proj('sturb', 2,data_source=disk_dd,weight_field='density')


    width = (float(L), 'kpc')
    NN=int(L/dr)
    dA=((L/NN)**2).in_units('pc**2')

    res = [NN, NN]
    frb = proj.to_frb(width, res, center=[0.5,0.5,0.5])

    print('### Turbulence Map ###')
    v3D=frb['sturb'].in_units('km/s')

    np.save(name+'_sigma_3D',v3D)
    return True

def Sigma_mass_3D(name):
    if os.path.isfile(name+'_sigma_mass_3D.npy'):
        return True
    data='Sims/'+name+'/G-'+name[-4:]
    ds=yt.load(data)

    L=40*kpc

    grids=ds.refine_by**ds.index.max_level*ds.domain_dimensions[0]

    DX=ds.arr(1, 'code_length')
    DX.convert_to_units('pc')
    dr=DX/grids

    dd=ds.all_data()
    print('### Projection ###')
    disk_dd = dd.cut_region(["obj['Disk_H'].in_units('pc') < 1.0e3"])
    proj = ds.proj('wturb', 2,data_source=disk_dd,weight_field='density')


    width = (float(L), 'kpc')
    NN=int(L/dr)
    dA=((L/NN)**2).in_units('pc**2')

    res = [NN, NN]
    frb = proj.to_frb(width, res, center=[0.5,0.5,0.5])

    print('### Turbulence Map ###')
    v3D=frb['wturb'].in_units('km/s')

    np.save(name+'_sigma_mass_3D',v3D)
    return True

def Sound_speed_map(name):
    if os.path.isfile(name+'_cs.npy'):
        return True
    data='Sims/'+name+'/G-'+name[-4:]
    ds=yt.load(data)

    L=40*kpc

    grids=ds.refine_by**ds.index.max_level*ds.domain_dimensions[0]

    DX=ds.arr(1, 'code_length')
    DX.convert_to_units('pc')
    dr=DX/grids

    dd=ds.all_data()
    print('### Projection ###')
    disk_dd = dd.cut_region(["obj['Disk_H'].in_units('pc') < 1.0e3"])
    proj = ds.proj('sound_speed', 2,data_source=disk_dd,weight_field='density')


    width = (float(L), 'kpc')
    NN=int(L/dr)
    dA=((L/NN)**2).in_units('pc**2')

    res = [NN, NN]
    frb = proj.to_frb(width, res, center=[0.5,0.5,0.5])

    cs=frb['sound_speed'].in_units('km/s')
    cs_2=np.sqrt(frb['sound_speed_2'].in_units('km**2/s**2'))
    cs_rep=1.0/frb['sound_speed_rep'].in_units('s/km')
    cs_rep_2=1.0/np.sqrt(frb['sound_speed_rep_2'].in_units('s**2/km**2'))

    np.save(name+'_cs',cs)
    np.save(name+'_cs_2',cs_2)
    np.save(name+'_cs_rep',cs_rep)
    np.save(name+'_cs_rep_2',cs_rep_2)

def Modify_Output(name,fields,factors,operations):
    """
    Allowed Enzo_Fields
    Density / Metal_Density / Temperature / TotalEnergy
    x-velocity / y-velocity / z-velocity
    """
    data='Sims/'+name+'/G-'+name[-4:]
    ds=yt.load(data)

    out_basename = 'Sims/ModifiedRestart/Mod_'+name
    if os.path.isdir(out_basename):
        pass
    else:
        os.system('mkdir '+out_basename)
    for  g in ds.index.grids:
        out_file_name = "%s/%s"%(out_basename , g.filename.split("/")[-1])

        in_cpu = h5py.File(g.filename,'r')
        in_group = in_cpu['Grid%08d'%g.id]
        out_cpu = h5py.File(out_file_name,'a')
        out_group = out_cpu.require_group( 'Grid%08d'%g.id )

        for in_field in in_group:
            out_group.require_dataset(name=in_field,shape=in_group[in_field].shape,dtype=in_group[in_field].dtype)
            this_array = in_group[in_field][:]
            if in_field in fields:
                loc=int(np.where(fields==in_field,1,0))
                factor=factors[loc]
                operation=operations[loc]
                if operation=='addition':
                    this_array += factor
                if operation=='product':
                    this_array *= factor
            out_group[in_field][:] = this_array
        out_cpu.close()
        in_cpu.close()
    lista=glob.glob('Sims/'+name+'/*')
    for li in lista:
        if 'cpu' not in li:
            dest="%s/"%(out_basename)
            os.system('cp '+li+' '+dest)
    for li in lista:
        if ('cpu' not in li)&('hdf' not in li):
            change_word_infile(li,name,'Mod_'+name)

def Integrated_vz(name,L,H):
    if os.path.isfile(name+'_vz_integrated.npy'):
        return True
    data='Sims/'+name+'/G-'+name[-4:]
    ds=yt.load(data)

    L=L*kpc
    H=H*1000
    grids=ds.refine_by**ds.index.max_level*ds.domain_dimensions[0]

    DX=ds.arr(1, 'code_length')
    DX.convert_to_units('pc')
    dr=DX/grids

    dd=ds.all_data()
    print('### Projection ###')
    disk_dd = dd.cut_region(["obj['Disk_H'].in_units('pc') < "+str(H)+""])
    proj = ds.proj('vz', 2,data_source=disk_dd,weight_field='density')


    width = (float(L), 'kpc')
    NN=int(L/dr)
    dA=((L/NN)**2).in_units('pc**2')

    res = [NN, NN]
    frb = proj.to_frb(width, res, center=[0.5,0.5,0.5])

    vz=frb['vz'].in_units('km/s')

    np.save(name+'_vz_integrated.npy',vz)
