from scipy.fftpack import fftfreq
from itertools import product
import numpy as np

def fft_sample(n_kpoints, lp_supercell,):
    n_kx, n_ky, n_kz = n_kpoints
    lx, ly, lz = lp_supercell
    kx = fftfreq(n_kx, lx/n_kx)*2*np.pi
    ky = fftfreq(n_ky, ly/n_ky)*2*np.pi
    kz = fftfreq(n_kz, lz/n_kz)*2*np.pi
    
    kxyz = np.array(list(product(kx, ky, kz)))
    
    return kxyz

def ijkl_invert(sij_voigt, comb_dict):
    
    sijkl = np.zeros((3,3,3,3))
    for i, j, k, l in product(range(3), repeat=4):
        ij_ind, kl_idn = comb_dict[(i,j)], comb_dict[(k,l)]
        
        if ij_ind < 3 and kl_idn < 3:
            sijkl[i,j,k,l] = sij_voigt[ij_ind, kl_idn]
        elif np.min([ij_ind, kl_idn]) < 3 and np.max([ij_ind, kl_idn]) >= 3:
            sijkl[i,j,k,l] = sij_voigt[ij_ind, kl_idn]/2
        elif ij_ind >= 3 and kl_idn >= 3: 
            sijkl[i,j,k,l] = sij_voigt[ij_ind, kl_idn]/4
            
    return sijkl

def ijkl2voigt(sijkl, comb_dict):
    sij_voigt = np.zeros((6,6))
    for i,j,k,l in product(range(3), repeat=4):
        ij_ind, kl_ind = comb_dict[(i,j)], comb_dict[(k,l)]
        sij_voigt[ij_ind, kl_ind] += sijkl[i,j,k,l]
    return sij_voigt

def cij_rot(sijkl_raw, raw_lattice_vec, new_lattice_vec, comb_dict):

    nmt_raw = raw_lattice_vec/np.linalg.norm(raw_lattice_vec, axis=1)[:,None]
    rot_tensor = new_lattice_vec@np.linalg.inv(nmt_raw)
    
    sij_1100_ijkl = np.zeros((3,3,3,3))
    for i, j, k, l in product(range(3), repeat=4):
        for m, n, o, p in product(range(3), repeat=4):
            
            aim = rot_tensor[i, m]
            ajn = rot_tensor[j, n]
            ako = rot_tensor[k, o]
            alp = rot_tensor[l, p]
            
            sij_1100_ijkl[i, j, k, l] += aim*ajn*ako*alp*sijkl_raw[m, n, o, p]
    
    sij_1100 = ijkl2voigt(sij_1100_ijkl, comb_dict)
    cij_1100 = np.linalg.inv(sij_1100)

    return cij_1100


def ft_main_vec(x, *coef, n_coef=2, b=0,
                grad_sign_list=None):
    #* x, y should be normalized by a / c; coef as matrix input
    #* displacement in 1d array form
    x = x*2
    coef = np.array(coef)
    
    if coef.ndim == 2:
        c_init = coef[:,0]
        n = (coef.shape[1]-1)//n_coef
    elif coef.ndim == 1:
        c_init = coef[0]
        n= (len(coef)-1)//n_coef
        
    prefac = 1/2

    ij_vecbuffer = np.array(list(product(range(1, n+1), range(1,n_coef//2+1))))
    j_vec_, i_vec = ij_vecbuffer[:,1], ij_vecbuffer[:,0]
    j_vec = 2*j_vec_
    coefind_vec_sin = (n_coef*i_vec - j_vec + 1).astype(int) 
    coefind_vec_cos = (n_coef*i_vec - j_vec + 2).astype(int)

    if coef.ndim == 2:
        coef_vec_1 = coef[:,coefind_vec_sin]
        coef_vec_2 = coef[:,coefind_vec_cos]
    elif coef.ndim == 1:
        coef_vec_1 = coef[coefind_vec_sin]
        coef_vec_2 = coef[coefind_vec_cos]
    
    sin_x = np.sin(prefac**j_vec_*i_vec*np.pi*x.reshape(-1,1))
    cos_x = np.cos(prefac**j_vec_*i_vec*np.pi*x.reshape(-1,1))
    
    y_ft = c_init + np.sum(coef_vec_1*sin_x + coef_vec_2*cos_x, axis=1)

    if grad_sign_list is not None:
        y_ft = y_ft*grad_sign_list
        
    return y_ft

def second_derive(f, dx):
    len_f = len(f)
    #* padding for 0 and -1 based on PBC
    f_pad = np.concatenate((f[-1:], f, f[:1]), axis=0)
    f_plus = f_pad[2:len_f+2]
    f_minus = f_pad[:len_f]
    return (f_plus - 2*f + f_minus)/(dx**2)

def supercell_make(frac_coord, supercell_dim):
    nx, ny, nz = supercell_dim
    frac_raw, frac_n = frac_coord.copy(), frac_coord.copy()
    
    for i, j, k in product(range(nx), range(ny), range(nz)):
        if i== 0 and j == 0 and k == 0:
            continue
        
        frac_n = np.concatenate(
            (frac_n, frac_raw + np.array([i,j,k])), axis=0)
        
    return frac_n