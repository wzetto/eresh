import numpy as np
from itertools import product
from scipy.fftpack import fft2, ifft2, fftfreq
from multiprocessing import Pool

from scipy.optimize import curve_fit

from eresh.constants import MJsM2EVA
from eresh import utils

def u_x(x, disso_x, wx, b):
    #* initial trial function for x-axis displacement field
    return b/np.pi/2*(
        # np.arctan((x-1/2*np.max(x)-disso_x/2)/wx) + 
        # np.arctan((x-1/2*np.max(x)+disso_x/2)/wx)
        np.arctan((x-disso_x/2)/wx) + 
        np.arctan((x+disso_x/2)/wx)
    ) + b/2

def u_x_mov(x_, disso_x, wx, b, mov_x=0):
    '''
    b13 part
    '''
    x = x_ - mov_x #* move the x to center
    return b/np.pi/2*(
        np.arctan((x-disso_x/2)/wx) + 
        np.arctan((x+disso_x/2)/wx)
    ) + b/2
    
def e_ext(tau, ux):
    return np.sum(tau*ux)

#TODO type 1: energy minimization
def e_total(ux, tau, zero_k_ind, lattice_mesh,
            fmn_mesh,
            gsfe_coef, n_coef, b, #* gsfe part
            homog_strain = False,
            mode = 'get_parts'
            ):

    lattice_dim = lattice_mesh.shape
    #* define boundary zone
    bd_x = lattice_dim[0]//20
    
    #* boundary condition, otherwise fail to converge
    ux = ux.reshape(lattice_dim[0], lattice_dim[1])
    ny = lattice_dim[1]
    nx = lattice_dim[0]
    #* add boundary condition
    ux[0] = np.ones(ny)*0 
    ux[-1] = np.ones(ny)*b
    # ux[-1] = np.ones(ny)*max_ux
    # ux[0] = np.ones(ny)*min_ux
    
    #* mirroring to ensure PBC
    mirro_ind = np.arange(nx)[::-1]
    ux_mirro = ux[mirro_ind]
    
    #* elastic 
    # substract linear function to ensure PBC?
    # linear_func = ux_linearsubs(lattice_mesh[:,:,0], (max_ux-min_ux), lx).reshape(-1,1)
    # linear_func_tile = np.tile(linear_func, (1, lattice_dim[1]))
    k_ux = fft2(np.concatenate((ux, ux_mirro), axis=0))
    stress_k = fmn_mesh*k_ux
    #* FT(sigma)(0, 0) = 0
    stress_k[zero_k_ind[0]][zero_k_ind[1]] = np.array(0).astype(complex)
    
    #* use Parseval's identity
    e_el_ = np.real(np.sum(stress_k*np.conj(k_ux))/lattice_dim[0]/lattice_dim[1]/2)/4
    
    #TODO direct in r-space
    stress_r = ifft2(stress_k)
    # e_el_ = np.sum(np.real(stress_r[:nx])*(ux+dux))/2
    
    ux_ravel = (ux).flatten()
    ux_frac = (ux_ravel/b) % 1
    ux_frac[ux_frac > 1/2] = 1 - ux_frac[ux_frac > 1/2]
    
    #* misfit
    gamma_xy = utils.ft_main_vec(ux_frac, *gsfe_coef, n_coef=n_coef, )*MJsM2EVA
    e_ms_ = np.sum(gamma_xy)
    
    #* external stress 
    e_tau_ = e_ext(tau, ux_ravel)
    #TODO if apply homogeneous strain load
    # if homog_strain:
    #     e_tau_ += 1/2*sijkl[0,2,0,2]*tau**2*nx*ny*(a*np.sqrt(3))
    
    #* penalty term for minus gradient?
    # grad_bd = np.concatenate((np.gradient(ux[:bd_x], axis=0), np.gradient(ux[-bd_x:], axis=0)), axis=0)
    # grad_minus = np.sum(grad_bd[grad_bd < 0])
    #* penalty coef
    # e_alpha = 1e3
    # e_penalty = e_alpha*grad_minus
    e_penalty = 0
    
    #TODO displacement field gradient term 
    grad_epsilon = 0
    e_grad = grad_epsilon*np.sum(np.gradient(ux, axis=0)**2)
    
    if mode == 'get_parts':
        return stress_k, stress_r, e_ms_, e_el_, e_tau_, e_grad
    else:
        return e_ms_ + e_el_ + e_tau_ - e_penalty + e_grad

def jac_total(
            ux, tau, zero_k_ind, lattice_mesh,
            fmn_mesh,
            gsfe_coef_grad, n_coef_grad, b, #* gsfe part
            homog_strain = False,
            mode = 'get_parts'
        ):

    ''' 
    return jacobian as tau_13 + div(gamma)
    note that internal stress is directly calculated without mirroring
    '''
    
    lattice_dim = lattice_mesh.shape
    
    #* boundary condition, otherwise fail to converge
    ux = ux.reshape(lattice_dim[0], lattice_dim[1])
    
    ny = lattice_dim[1]
    nx = lattice_dim[0]
    #* add boundary condition
    ux[0] = np.ones(ny)*0 
    ux[-1] = np.ones(ny)*b
    
    #* mirroring to ensure PBC
    mirro_ind = np.arange(nx)[::-1]
    ux_mirro = ux[mirro_ind]

    k_ux = fft2(np.concatenate((ux, ux_mirro), axis=0))
    stress_k = fmn_mesh*k_ux
    #* FT(sigma)(0, 0) = 0
    stress_k[zero_k_ind[0]][zero_k_ind[1]] = np.array(0).astype(complex)
    stress_r = ifft2(stress_k)[:nx, :ny]
    
    #* gamma surface part (eV/Å^3)
    ux_frac = (ux/b) % 1
    grad_sign_list = np.sign(1/2-ux_frac).flatten()
    ux_frac[ux_frac > 1/2] = 1 - ux_frac[ux_frac > 1/2]
    
    d_gamma_xy = utils.ft_main_vec(ux_frac.flatten(), *gsfe_coef_grad, 
                             n_coef=n_coef_grad, b=b, 
                             grad_sign_list=grad_sign_list)
    
    stress_r = np.real(stress_r).flatten()
    d_gamma_xy = d_gamma_xy.flatten()
    
    if mode == 'get_parts':
        return stress_r, d_gamma_xy
    
    grad = stress_r + d_gamma_xy + tau
    
    return grad

def supercell_2d(frac, nx, ny):
    frac_raw, frac_n = frac.copy(), np.empty((0,2))
    nx_half, ny_half = nx//2, ny//2
    for i, j in product(range(-nx_half, nx_half+1), range(-ny_half, ny_half+1)):
        frac_n = np.concatenate((frac_n, frac_raw + np.array([i,j])))
    return frac_n

def adjoint_mat(x):
    return (np.linalg.det(x)*np.linalg.inv(x))

def d_kxkykz(cij, k_vec, ijkl2ij_dict):
    d_mat = np.zeros((3,3), dtype=complex)
    for i, j in product(range(3), repeat=2):
        for m, n in product(range(3), repeat=2):
            im_denote, jn_denote = ijkl2ij_dict[(i,m)], ijkl2ij_dict[(j,n)]
            km, kn = k_vec[m], k_vec[n]
            d_mat[i,j] += cij[im_denote,jn_denote]*km*kn
            
    return d_mat

def permu_sign(permu, permu_dict):
    try:
        epsilon = permu_dict[permu]
    except:
        epsilon = 0
        
    return epsilon

def ux_linearsubs(x, b, lx):
    ''' 
    Add the displacement field with linear function to ensure PBC
    '''
    return b - b/lx*x

#* main function 
def gl_minimize(
            ux, 
            tau, 
            zero_k_ind, lattice_mesh,
            fmn_mesh,
            gsfe_coef, n_coef, 
            gsfe_coef_grad, n_coef_grad,
            b_13,
            homog_strain,
            mode, 
            m, #* drag coef.
            converg_tol, max_iter):
    
    # e_raw = e_total(ux, dux, tau, zero_k_ind, lattice_mesh,
    #                 fmn_mesh, lx,
    #                 gsfe_coef, n_coef, b, 
    #                 homog_strain,
    #                 mode)
    ux_n = ux.copy().flatten()
    for i in range(max_iter):
        
        grad = jac_total(ux_n, tau, 
                        zero_k_ind, lattice_mesh,
                        fmn_mesh,
                        gsfe_coef_grad, n_coef_grad, b_13, 
                        homog_strain, mode)
        
        ux_n = ux_n - m*grad
        
        if np.linalg.norm(grad) < converg_tol:
            break 
        
        #! warning for max iteration
        # if i == max_iter-1:
        #     print(f'Warning: maximum iteration reached, grad norm = {np.linalg.norm(grad)}')
        
        # e_raw = e
    
    e_optim = e_total(ux_n, tau, 
                    zero_k_ind, lattice_mesh,
                    fmn_mesh,
                    gsfe_coef, n_coef, b_13, 
                    homog_strain,
                    mode)
       
    return ux_n, e_optim

class pfdd:
    def __init__(self,
                 mjm2eva,
                 eva2gpa,
                 deform_mode,
                 specie_denote,
                 nx, ny,
                 lattice_param,
                 b_norm,
                 stiffness_tensor_buffer,
                 gsfe_coef,
                 gsfe_grad_coef,
                 gsfe_ncoef,
                 pn_kernel_buffer,
                 pn_derive_thread):
        
        self.mjm2eva = mjm2eva
        self.eva2gpa = eva2gpa
        
        if deform_mode == 'edge':
            self.deform_mode = 'f13'
        elif deform_mode == 'screw':
            self.deform_mode = 'g23'
            
        self.specie_denote = specie_denote
        self.nx = nx
        self.ny = ny
        self.lattice_param = lattice_param
        self.b_norm = b_norm
        self.stiffness_tensor_buffer = stiffness_tensor_buffer
        self.gsfe_coef = gsfe_coef
        self.gsfe_grad_coef = gsfe_grad_coef
        self.gsfe_ncoef = gsfe_ncoef
        self.pn_kernel_buffer = pn_kernel_buffer
        self.pn_derive_thread = pn_derive_thread
        self.lx, self.ly = b_norm*nx, b_norm*ny
        self.half_nx, self.half_ny = nx//2, ny//2
    
    def lattice_grid_prepare(self,):
        len_nx, len_ny = self.nx, self.ny
        lattice_grid_2d = np.zeros((len_nx, len_ny, 2))
        half_nx, half_ny = len_nx//2, len_ny//2
        for i, j in product(range(-half_nx, half_nx+1), range(-half_ny, half_ny+1)):
            #* x-only displacement field from linear assumption
            lattice_grid_2d[i+half_nx,j+half_ny] = np.array([i*self.b_norm, j*self.b_norm])
            
        return lattice_grid_2d
    
    def _zero_k_ind(self,):
        len_kx, len_ky = self.nx*2, self.ny
        kx, ky = fftfreq(len_kx, self.lx/self.nx)*2*np.pi, fftfreq(len_ky, self.ly/self.ny)*2*np.pi
        #* find zero index (set self-stress as zero)
        zero_kx = np.where(np.abs(kx) < 1e-10)[0][0]
        zero_ky = np.where(np.abs(ky) < 1e-10)[0][0]
        return [zero_kx, zero_ky]
    
    def u_x_fixb(self, x, disso_x, wx):
        #* initial trial function for x-axis displacement field
        return self.b_norm/np.pi/2*(
            # np.arctan((x-1/2*np.max(x)-disso_x/2)/wx) + 
            # np.arctan((x-1/2*np.max(x)+disso_x/2)/wx)
            np.arctan((x-disso_x/2)/wx) + 
            np.arctan((x+disso_x/2)/wx)
        ) + self.b_norm/2
    
    def optimize(self, tau_trial, ux_grid, zero_k_ind,
                 fmn_meshgrid,):
        
        energy_info = []
        ux_info = []
        tau = tau_trial #* external stress
        lattice_grid_2d = self.lattice_grid_prepare()
            
        ux_opt, e_opt = gl_minimize(
            ux_grid.flatten(), 
            tau,
            zero_k_ind, lattice_grid_2d,
            fmn_meshgrid,
            self.gsfe_coef, self.gsfe_ncoef, 
            self.gsfe_grad_coef, self.gsfe_ncoef,
            self.b_norm,
            False,
            'None',
            m=1e-1,
            #TODO convergence criteria
            converg_tol=1e-6,
            max_iter=1000,
        )
        
        ux_info.append(ux_opt)
        energy_info.append(e_opt)
                
        return ux_info, energy_info
    
    def disfield_cal(self,):
        #* initialize dissociation distance for prism <a+c> dislocation
        d_list = np.linspace(20, 60, 21)
    
        e_ux_info_buffer = {}
        for specie_name in self.stiffness_tensor_buffer.keys():
            tau_list = []
            kernel_res = self.pn_kernel_buffer[specie_name][self.deform_mode]
            for d_init in d_list:
                ux_grid_2d = np.zeros((self.nx, self.ny))
                for i, j in product(range(-self.half_nx, self.half_nx+1), range(-self.half_ny, self.half_ny+1)):
                    #TODO x-only displacement field from linear assumption
                    ux_grid_2d[i+self.half_nx,j+self.half_ny] = \
                        u_x(i*self.b_norm, d_init, 1, self.b_norm)
                    
                tau_list.append((0., ux_grid_2d, self._zero_k_ind(), kernel_res))
                
            # p = Pool(min(len(tau_list), self.pn_derive_thread))
            # e_ux_info_specie = p.starmap(self.optimize, tau_list)
            # p.close()
            
            #! use for-loop
            e_ux_info_specie = []
            for tau_input in tau_list:
                e_ux_info_specie.append(self.optimize(*tau_input))
            
            ux_list = [np.array(r[0]) for r in e_ux_info_specie]
            e_list = [r[1] for r in e_ux_info_specie]
            e_ux_info_buffer[specie_name] = {
                'ux': ux_list,
                'e': e_list,
            }
        
        return e_ux_info_buffer
    
    def ux_fit(self, ux_input):
        optim_ux_raw = ux_input.copy()
        optim_1d = optim_ux_raw.reshape((self.nx, self.ny))[:,0]
        grid_x = np.arange(-self.half_nx, self.half_nx+1)*self.b_norm

        params = curve_fit(self.u_x_fixb, grid_x, optim_1d)
        disso_x, wx = params[0]
        
        return disso_x, wx
    
    def heb_optimal(self, e_ux_info_buffer):
        heb_info = e_ux_info_buffer[self.specie_denote]
        ux_optimal = heb_info['ux'][np.argmin(heb_info['e'])]
        heb_opt_d, heb_opt_w = self.ux_fit(ux_optimal)
        
        return heb_opt_d, heb_opt_w

    def heb_ele_optimal(self):
        
        e_ux_info_buffer = self.disfield_cal()  
        heb_opt_d_w = self.heb_optimal(e_ux_info_buffer)
        grid_x = np.arange(-self.half_nx, self.half_nx+1)*self.b_norm
        d_w_ele_buffer = {}
        for key in e_ux_info_buffer.keys():
            if key == self.specie_denote:
                u0_heb = e_ux_info_buffer[key]['ux'][np.argmin(e_ux_info_buffer[key]['e'])]
                u0_heb = u0_heb.reshape((self.nx, self.ny))[:,0]
                ufin_heb = np.zeros_like(u0_heb)
                ufin_heb[1:] = u0_heb[:-1]
            
                d_w_ele_buffer[key] = {
                    'd_optimal': heb_opt_d_w[0],
                    'w_optimal': heb_opt_d_w[1],
                    'e_optimal': min(e_ux_info_buffer[key]['e']),
                    'ux_opt': u0_heb,
                    'ux_opt_mov': ufin_heb,
                    'grid_x': grid_x,
                }
                continue
            
            ux_ele, e_ele = e_ux_info_buffer[key]['ux'], e_ux_info_buffer[key]['e']
            ux_ele_d_list = [self.ux_fit(ux)[0] for ux in ux_ele]
            ux_ele_w_list = [self.ux_fit(ux)[1] for ux in ux_ele]

            ux_ele_opt_ind = np.argmin(
                np.abs(np.array(ux_ele_d_list) - heb_opt_d_w[0])
            )
            #* move displacement field to create next equilibrium state
            u0 = ux_ele[ux_ele_opt_ind].reshape((self.nx, self.ny))[:,0]
            ufin = np.zeros_like(u0)
            ufin[1:] = u0[:-1]
            d_w_ele_buffer[key] = {
                                   'd_optimal': ux_ele_d_list[ux_ele_opt_ind],
                                   'w_optimal': ux_ele_w_list[ux_ele_opt_ind],
                                   'e_optimal': e_ele[ux_ele_opt_ind],
                                   'ux_opt': u0,
                                   'ux_opt_mov': ufin,
                                   'grid_x': grid_x,}
        
        
        return d_w_ele_buffer, heb_opt_d_w