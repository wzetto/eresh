import numpy as np
from itertools import product
from scipy.integrate import quad
import pickle
from scipy.interpolate import CubicSpline 

from eresh.constants import VOIGT_DICT 

def d_derive(t):
    i = np.eye(3)
    cross_t = np.outer(t, t)
    return i - cross_t

def v_vec_derive(d):
    v = np.zeros(3)
    for j in range(3):
        v[j] = np.sum([d[i,j]**2 for i in range(3)])**0.5   
        
    return v

def mij_derive(m, n, cij):
    mn_mat = np.zeros((3, 3))
    for i, j, k, l in product(range(3), repeat=4):
        mn_mat[j,k] += cij[VOIGT_DICT[(i,j)], VOIGT_DICT[(k,l)]] * m[i] * n[l]

    return mn_mat

def bij_mat_part_derive(omega, cij, m_vec, n_vec, i, j):
    
    m = np.cos(omega)*m_vec + np.sin(omega)*n_vec
    n = -np.sin(omega)*m_vec + np.cos(omega)*n_vec
    
    mm_mat = mij_derive(m, m, cij)
    mn_mat = mij_derive(m, n, cij)
    nm_mat = mij_derive(n, m, cij)
    nn_mat = mij_derive(n, n, cij) + 1e-10
    nn_mat_inv = np.linalg.inv(nn_mat)
    
    bij_mat_part = 1/8/np.pi**2*mm_mat[i,j]
    
    for k, l in product(range(3), repeat=2):
        bij_mat_part += 1/8/np.pi**2* \
            (- mn_mat[i,l]*nn_mat_inv[l,k]*nm_mat[k,j])
            
    return bij_mat_part

def line_vec(a, c, vec_specify = None, disloc_mode = None):
    if vec_specify is None and disloc_mode is None:
        raise ValueError("Either vec_specify or disloc_mode must be provided.")
    elif disloc_mode == 'edge':
        return np.array([-c, a, 0])
    elif disloc_mode == 'screw':
        return np.array([a, c, 0])
    return vec_specify

def rot_mat_2d(theta):
    #* rotation direction is counter-clockwise
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

def second_derive(f, dx):
    len_f = len(f)
    #* padding for 0 and -1 based on PBC
    f_pad = np.concatenate((f[-1:], f, f[:1]), axis=0)
    f_plus = f_pad[2:len_f+2]
    f_minus = f_pad[:len_f]
    return (f_plus - 2*f + f_minus)/(dx**2)
    
class lt_prefactor:
    def __init__(self,
                stiffness_tensor_eva3,
                b_norm,
                b_partial_vec,
                sav_pth=None,):
        
        self.cij = stiffness_tensor_eva3
        assert np.max(self.cij) < 10, "Maximum modulus > 1600 GPa, check input stiffness tensor."
        self.b_norm = b_norm
        self.b_partial_vec = b_partial_vec
        self.sav_pth = sav_pth
        
    def __call__(self,):
        
        b1 = self.b_partial_vec
        b2 = self.b_partial_vec
        b_full_vec = b1 + b2
            
        vec_init = b_full_vec/self.b_norm #* inital dislocation line direction

        rot_theta_buffer = np.linspace(0, np.pi, 1000)
        
        prefac_list = []
        for rot_theta in rot_theta_buffer:
            t = np.zeros_like(vec_init)
            t[:2] = rot_mat_2d(rot_theta) @ vec_init[:2]
            t = t / np.linalg.norm(t)

            d_mat = d_derive(t)
            v_vec = v_vec_derive(d_mat)
            alpha = np.argmax(v_vec)
            d_alpha = d_mat[:, alpha]
            n_vec = d_alpha / np.linalg.norm(d_alpha)
            m_vec = np.cross(n_vec, t)

            b_mat = np.zeros((3, 3))
            for i, j in product(range(3), repeat=2):
                b_mat[i,j] = quad(bij_mat_part_derive, 0, np.pi*2, 
                                args=(self.cij, m_vec, n_vec, i, j))[0]

            prefac = 0
            for i, j in product(range(3), repeat=2):
                prefac += b1[i] * b2[j] * b_mat[i, j]
                
            prefac_list.append(prefac)

        prefac_list = np.array(prefac_list) 
        second_k = second_derive(prefac_list[:-1], rot_theta_buffer[1] - rot_theta_buffer[0])
        second_k = np.concatenate((second_k, [second_k[0]]))  #* padding for PBC

        cs = CubicSpline(rot_theta_buffer, prefac_list + second_k, bc_type='periodic')
        
        if self.sav_pth is not None:
            with open(self.sav_pth, 'wb') as f:
                pickle.dump(cs, f)
            print(f'line energy prefactor saved to {self.sav_pth}')
        
        return cs