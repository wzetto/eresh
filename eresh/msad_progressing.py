import numpy as np
import pickle
import sympy as sp
import multiprocessing as mp    
import os

#! under development 

def k_eval(lambda_k, kxyz):
    kx, ky, kz = kxyz 
    return np.abs(lambda_k)**2*np.exp(-a**2*kx**2-c**2*ky**2-3*a**2*kz**2)

def msad_ele(eigen_ele, v_tmb2, v_lattice, kernel_result, 
             n_sites, c_ele, structure_fac):
    
    k_term = n_sites*c_ele*(1-c_ele) + c_ele**2*np.abs(structure_fac)**2
    k_term = np.sum(k_term*kernel_result)
    return (eigen_ele*v_tmb2/v_lattice)**2*k_term

specie_denote = 'TiVZrB2'
superfast = True #TODO if turn on superfast mode 
if superfast:
    calc_denote = '_superfast'
else:
    calc_denote = ''
compo_space_denote = '_tivzrnbmohftawb2'

a, c = 3.09,3.28 #TODO lattice parameters
ti, v, zr, nb, mo, hf, ta, w = 1/3, 1/3, 1/3, 0., 0, 0, 0, 0 #TODO composition
smear_ti, smear_v, smear_zr, smear_nb, smear_mo, smear_hf, smear_ta, smear_w = 1,1,1,1,1,1,1,1 #TODO smearing coefficients
nx, ny, nz = 64,64,64
nxyz = nx*ny*nz
eigenstrain_mode_list = ['a', 'c', 'a_sqrt3']
lattice_axis_list = ['1', '2', '3']

#* load kmesh
kernel_savpth = f'kernel/runs/{specie_denote}/{specie_denote}_{a}_{c}_{nx}_{ny}_{nz}'
kmesh = np.load(f'{kernel_savpth}_kmesh{calc_denote}{compo_space_denote}.npy')

#* derive rmesh
# rx = np.linspace(0, a*nx, nx, endpoint=False)
# ry = np.linspace(0, c*ny, ny, endpoint=False)
# rz = np.linspace(0, a*np.sqrt(3)*nz, nz, endpoint=False)
# rxyz = np.array(list(product(rx, ry, rz)))
#* derive structural factor for each k
# sk_vec = np.sum(np.exp(1j*np.dot(kmesh, rxyz.T)), axis=1)

#* lattice vector and cluster volume
lattice_vec = np.array([
    [a, 0, 0],
    [0, c, 0],
    [0, 0, a*np.sqrt(3)]
])
lattice_vol = np.linalg.det(lattice_vec)
v_tmb2 = lattice_vol/2
lattice_vol_tot = lattice_vol*nx*ny*nz

msd_dict = {}
tot_msd = 0
var_eige_tot = 0
xti, xv, xzr, xnb, xmo, xhf, xta, xw = sp.symbols('TiB_2 VB_2 ZrB_2 NbB_2 MoB_2 HfB_2 TaB_2 WB_2')
for eigenstrain_mode, lattice_axis in zip(eigenstrain_mode_list, lattice_axis_list):

    #* load interaction kernel
    prefac = np.load(f'{kernel_savpth}_lambda{lattice_axis}{calc_denote}{compo_space_denote}.npy')

    #* load eigenstrain
    eigen_express = pickle.load(open(f'expression/eigenstrain/eigenstrain_{eigenstrain_mode}{calc_denote}{compo_space_denote}.pkl', 'rb'))
    eigen_ti, eigen_v, eigen_zr, eigen_nb, eigen_mo, eigen_hf, eigen_ta, eigen_w = eigen_express
    ti_eigen = eigen_ti.evalf(subs={xti:ti, xv:v, xzr:zr, xnb:nb, xmo:mo, xhf:hf, xta:ta, xw:w})
    v_eigen = eigen_v.evalf(subs={xti:ti, xv:v, xzr:zr, xnb:nb, xmo:mo, xhf:hf, xta:ta, xw:w})
    zr_eigen = eigen_zr.evalf(subs={xti:ti, xv:v, xzr:zr, xnb:nb, xmo:mo, xhf:hf, xta:ta, xw:w})
    nb_eigen = eigen_nb.evalf(subs={xti:ti, xv:v, xzr:zr, xnb:nb, xmo:mo, xhf:hf, xta:ta, xw:w})
    mo_eigen = eigen_mo.evalf(subs={xti:ti, xv:v, xzr:zr, xnb:nb, xmo:mo, xhf:hf, xta:ta, xw:w})
    hf_eigen = eigen_hf.evalf(subs={xti:ti, xv:v, xzr:zr, xnb:nb, xmo:mo, xhf:hf, xta:ta, xw:w})
    ta_eigen = eigen_ta.evalf(subs={xti:ti, xv:v, xzr:zr, xnb:nb, xmo:mo, xhf:hf, xta:ta, xw:w})
    w_eigen = eigen_w.evalf(subs={xti:ti, xv:v, xzr:zr, xnb:nb, xmo:mo, xhf:hf, xta:ta, xw:w})
    
    print(zr_eigen, mo_eigen)
    
    var_eigenstrain = smear_ti*ti*ti_eigen**2 + \
                    smear_v*v*v_eigen**2 + \
                    smear_zr*zr*zr_eigen**2 + \
                    smear_nb*nb*nb_eigen**2 + \
                    smear_mo*mo*mo_eigen**2 + \
                    smear_hf*hf*hf_eigen**2 + \
                    smear_ta*ta*ta_eigen**2 + \
                    smear_w*w*w_eigen**2   
    var_eige_tot += var_eigenstrain

    #* old ver. (Ti-V-Zr)B2 only but higher precision
    # eigen_express = pickle.load(open(f'/home/wang/Documents/HEB/pf_distort/msad_microelasticity/expression/eigenstrain/eigenstrain_{eigenstrain_mode}.pkl', 'rb'))
    # eigen_ti, eigen_v, eigen_zr = eigen_express
    # ti_eigen = eigen_ti.evalf(subs={xti:ti, xv:v, xzr:zr})
    # v_eigen = eigen_ti.evalf(subs={xti:ti, xv:v, xzr:zr})
    # zr_eigen = eigen_zr.evalf(subs={xti:ti, xv:v, xzr:zr})
    # var_eigenstrain = ti*ti_eigen**2 + v*v_eigen**2 + zr*zr_eigen**2
    
    #* evaluate the displacement's variance per axis
    p = mp.Pool(54)
    kernel_result = p.starmap(k_eval, zip(prefac, kmesh))

    # msd_ti = msad_ele(ti_eigen, v_tmb2, lattice_vol_tot, kernel_result, nxyz, ti, sk_vec)
    # msd_v = msad_ele(v_eigen, v_tmb2, lattice_vol_tot, kernel_result, nxyz, v, sk_vec)
    # msd_zr = msad_ele(zr_eigen, v_tmb2, lattice_vol_tot, kernel_result, nxyz, zr, sk_vec)
    # msd_mo = msad_ele(mo_eigen, v_tmb2, lattice_vol_tot, kernel_result, nxyz, mo, sk_vec)
    # msd_hf = msad_ele(hf_eigen, v_tmb2, lattice_vol_tot, kernel_result, nxyz, hf, sk_vec)
    # msd_w = msad_ele(w_eigen, v_tmb2, lattice_vol_tot, kernel_result, nxyz, w, sk_vec)
    # print(f'{lattice_axis} ti: {msd_ti:.2f}, v: {msd_v:.2f}, zr: {msd_zr:.2f}, mo: {msd_mo:.2f}, hf: {msd_hf:.2f}, w: {msd_w:.2f}')
    
    msd = v_tmb2*var_eigenstrain/lattice_vol_tot*np.sum(kernel_result)*1e4 #* in pm^2
    msd_dict[f'{lattice_axis}_microEmodel'] = msd
    msd_dict[f'{lattice_axis}_ti_vareigstrain'] = ti*ti_eigen**2
    msd_dict[f'{lattice_axis}_v_vareigstrain'] = v*v_eigen**2
    msd_dict[f'{lattice_axis}_zr_vareigstrain'] = zr*zr_eigen**2
    msd_dict[f'{lattice_axis}_nb_vareigstrain'] = nb*nb_eigen**2
    msd_dict[f'{lattice_axis}_mo_vareigstrain'] = mo*mo_eigen**2
    msd_dict[f'{lattice_axis}_hf_vareigstrain'] = hf*hf_eigen**2
    msd_dict[f'{lattice_axis}_ta_vareigstrain'] = ta*ta_eigen**2
    msd_dict[f'{lattice_axis}_w_vareigstrain'] = w*w_eigen**2
    tot_msd += msd

sav_pth = f'runs/{specie_denote}/{specie_denote}_{a}_{c}_{nx}_{ny}_{nz}_msad{calc_denote}{compo_space_denote}.pkl'
create_dir(os.path.dirname(sav_pth))
pickle.dump(msd_dict, open(sav_pth, 'wb'))

msd_dict, tot_msd, var_eige_tot