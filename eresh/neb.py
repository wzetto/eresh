import numpy as np
from itertools import product
from scipy.fftpack import fft2, ifft2, fftfreq
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline

from eresh.constants import MJsM2EVA

def u_x_mov(x_, disso_x, wx, b, mov_x=0):
    '''
    b13 part
    '''
    x = x_ - mov_x #* move the x to center
    return b/np.pi/2*(
        np.arctan((x-disso_x/2)/wx) + 
        np.arctan((x+disso_x/2)/wx)
    ) + b/2
    
def tangent_vec_derive(u_ip1, u_im1, dim=2):
    if dim == 2:
        return (u_ip1-u_im1)/np.linalg.norm(u_ip1-u_im1, axis=1).reshape(-1, 1)
    elif dim == 1:
        return (u_ip1-u_im1)/np.linalg.norm(u_ip1-u_im1)

def f_pot(grad_e, u_ip1, u_im1):
    tangent_vec = tangent_vec_derive(u_ip1, u_im1)
    return -grad_e + np.sum(tangent_vec*grad_e, axis=1).reshape(-1,1) * tangent_vec

def f_spring(u_ip1, ui, u_im1, k_spring):
    tangent_vec = tangent_vec_derive(u_ip1, u_im1)
    u_part = (np.linalg.norm(u_ip1-ui, axis=1) - np.linalg.norm(u_im1-ui, axis=1)).reshape(-1, 1)
    return k_spring* u_part * tangent_vec

def f_saddle_derive(grad_e, tangent_vec):
    return -grad_e + 2* np.sum(tangent_vec*grad_e) * tangent_vec

def image_linearinterp(u_init, u_fin, n_img):
    if n_img < 2:
        raise ValueError("n_img must be at least 2")
    return np.array([u_init + (u_fin - u_init) * i / (n_img - 1) for i in range(n_img)])

def image_linearinterp_uxfunc(u_init, u_fin, n_img, dissox, wx, grid_x, b):
    img_buffer = []
    for i in range(n_img):
        if i == 0:
            img_buffer.append(u_init)
        elif i == n_img - 1:
            img_buffer.append(u_fin)
        else:
            mov_x = b/(n_img-1)*i
            u_i = u_x_mov(grid_x, dissox, wx, b, mov_x)
            img_buffer.append(u_i)
    return np.array(img_buffer)

def e_ext(tau, ux):
    return np.sum(tau*ux)

def redist(ux_buffer_):
    arc_loc = np.array(
        [np.sum(ux_buffer_[0]-ux_buffer_[i]) for i in range(len(ux_buffer_))]
    )
    
    arc_loc = arc_loc[np.argsort(arc_loc)]
    ux_buffer = ux_buffer_[np.argsort(arc_loc)].copy()
    
    #! remove negative arc length
    arc_loc_raw = arc_loc.copy()
    arc_loc = arc_loc[arc_loc_raw > -1e-3]
    ux_buffer = ux_buffer[arc_loc_raw > -1e-3]
    
    arc_loc_uniform = np.linspace(0, arc_loc_raw[-1], len(arc_loc_raw))
    ux_buffer_redist = np.zeros_like(ux_buffer_)
    for column_i in range(ux_buffer.shape[1]):
        cs = CubicSpline(arc_loc, ux_buffer[:, column_i])
        ux_buffer_redist[:, column_i] = cs(arc_loc_uniform)
        
    return ux_buffer_redist

def ft_main_vec(x, *coef, n_coef=2, print_=False, derivative_1st=False, b=0,
                grad_sign_list=None):
    #* x, y should be normalized by a / c; coef as matrix input
    x = x*2
    coef = np.array(coef)
    c_init = coef[0]
    n = (len(coef)-1)//n_coef
    
    prefac = 1/2

    ij_vecbuffer = np.array(list(product(range(1, n+1), range(1,n_coef//2+1))))
    j_vec_, i_vec = ij_vecbuffer[:,1], ij_vecbuffer[:,0]
    j_vec = 2*j_vec_
    coefind_vec_sin = (n_coef*i_vec - j_vec + 1).astype(int) 
    coefind_vec_cos = (n_coef*i_vec - j_vec + 2).astype(int)

    coef_vec_1 = coef[coefind_vec_sin]
    coef_vec_2 = coef[coefind_vec_cos]
    
    if not derivative_1st:
        
        #* assume x is a 1d array
        sin_x = np.sin(prefac**j_vec_*i_vec*np.pi*x.reshape(-1,1))
        cos_x = np.cos(prefac**j_vec_*i_vec*np.pi*x.reshape(-1,1))
        
        y_ft = c_init + np.sum(coef_vec_1*sin_x + coef_vec_2*cos_x, axis=1)
        
    elif derivative_1st:
        
        cos_x = 2/b*np.cos(prefac**j_vec_*i_vec*np.pi*x.reshape(-1,1))*(prefac**j_vec_*i_vec)*np.pi
        sin_x = 2/b*np.sin(prefac**j_vec_*i_vec*np.pi*x.reshape(-1,1))*(prefac**j_vec_*i_vec)*np.pi
        
        y_ft = np.sum(coef_vec_1*cos_x - coef_vec_2*sin_x, axis=1)

    if grad_sign_list is not None:
        y_ft = y_ft*grad_sign_list
        
    return y_ft

#TODO type 1: energy minimization
def e_total(ux, tau, zero_k_ind, lattice_mesh,
            fmn_mesh,
            gsfe_coef, n_coef, b, #* gsfe part
            mode = None
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
    
    #* mirroring to ensure PBC
    mirro_ind = np.arange(nx)[::-1]
    ux_mirro = ux[mirro_ind]
    
    #* elastic
    k_ux = fft2(np.concatenate((ux, ux_mirro), axis=0))
    stress_k = fmn_mesh*k_ux
    #* FT(sigma)(0, 0) = 0
    stress_k[zero_k_ind[0]][zero_k_ind[1]] = np.array(0).astype(complex)
    
    #TODO use Parseval's identity
    e_el_ = np.real(np.sum(stress_k*np.conj(k_ux))/lattice_dim[0]/lattice_dim[1]/2)/4
    
    #TODO direct in r-space
    stress_r = ifft2(stress_k)
    # e_el_ = np.sum(np.real(stress_r[:nx])*(ux+dux))/2
    
    ux_ravel = (ux).flatten()
    ux_frac = (ux_ravel/b) % 1
    ux_frac[ux_frac > 1/2] = 1 - ux_frac[ux_frac > 1/2]
    
    #* misfit
    gamma_xy = ft_main_vec(ux_frac, *gsfe_coef, n_coef=n_coef, print_=False)*MJsM2EVA
    e_ms_ = np.sum(gamma_xy)
    
    #* external stress 
    e_tau_ = e_ext(tau, ux_ravel)
    
    e_penalty = 0
    
    #TODO displacement field gradient term 
    grad_epsilon = 0
    e_grad = grad_epsilon*np.sum(np.gradient(ux, axis=0)**2)
    
    if mode == 'get_parts':
        return stress_k, stress_r, e_ms_, e_el_, e_tau_, e_grad
    else:
        return e_ms_ + e_el_ + e_tau_ - e_penalty + e_grad

def grad_e(
            ux, tau, zero_k_ind, lattice_mesh,
            fmn_mesh,
            gsfe_coef_grad, n_coef_grad, b, #* gsfe part
            mode = None
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
    
    d_gamma_xy = ft_main_vec(ux_frac.flatten(), *gsfe_coef_grad, 
                             n_coef=n_coef_grad, print_=False, 
                             derivative_1st=False, b=b, 
                             grad_sign_list=grad_sign_list)
    
    stress_r = np.real(stress_r).flatten()
    d_gamma_xy = d_gamma_xy.flatten()
    
    if mode == 'get_parts':
        return stress_r, d_gamma_xy
    
    grad = stress_r + d_gamma_xy + tau
    
    return grad

def neb_update(u_buffer_, n_iter, grad_coef, 
               zero_k_ind, lattice_mesh, fmn_mesh,
               gsfe_coef, n_coef, 
               gsfe_coef_grad, n_coef_grad, b,
               tol = 1e-5, k_spring = 0.1,
               redist_freq = 25):
    
    u_buffer = u_buffer_.copy()
    f_norm_list = []
    u0, ufin = u_buffer_[0], u_buffer_[-1]
    for i in range(n_iter):
        u_ip1 = u_buffer[2:]
        u_im1 = u_buffer[:-2]
        u_i = u_buffer[1:-1]
        e_val = np.array([e_total(u, 0, zero_k_ind, lattice_mesh, 
                         fmn_mesh, gsfe_coef, n_coef, b) for u in u_buffer]).reshape(-1,1)
        # print(e_val.shape)
        grad_e_val = np.array([grad_e(u, 0, zero_k_ind, lattice_mesh,
                             fmn_mesh, gsfe_coef_grad, n_coef_grad, b) for u in u_buffer[1:-1]])
        # print(grad_e_val.shape)
        f_buffer = f_pot(grad_e_val, u_ip1, u_im1) + f_spring(u_ip1, u_i, u_im1, k_spring)
        
        #! turn on saddle point correction
        saddle_pt = np.argmax(e_val.flatten()[1:-1])
        f_saddle = f_saddle_derive(grad_e_val[saddle_pt], tangent_vec_derive(u_ip1[saddle_pt], u_im1[saddle_pt], dim=1))
        f_buffer[saddle_pt] = f_saddle
        
        u_buffer[1:-1] += grad_coef * f_buffer
        
        f_norm = np.linalg.norm(f_buffer, axis=1)
        if np.all(f_norm < tol):
            print(f"Converged after {i} iterations.")
            break
        
        f_norm_list.append(f_norm)
        # if i % 100 == 0:
        #     clear_output(wait=True)
        #     plt.title(f'max norm {np.max(f_norm):.8f}')
        #     plt.plot(f_norm_list, alpha=0.3)
        #     plt.show()
            
        #* redistribute the displacement field as equal spacing on hyperarc
        if i % redist_freq == 0:
            u_buffer = redist(u_buffer)
            u_buffer[0] = u0.copy()
            u_buffer[-1] = ufin.copy()
        
    return u_buffer, e_val

class neb:
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
                 pn_derive_thread,
                 
                 element_info_buffer,
                 n_image):
        
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
        
        self.n_image = n_image
        self.element_info_buffer = element_info_buffer
        
    def _zero_k_ind(self,):
        len_kx, len_ky = self.nx*2, self.ny
        kx, ky = fftfreq(len_kx, self.lx/self.nx)*2*np.pi, fftfreq(len_ky, self.ly/self.ny)*2*np.pi
        #* find zero index (set self-stress as zero)
        zero_kx = np.where(np.abs(kx) < 1e-10)[0][0]
        zero_ky = np.where(np.abs(ky) < 1e-10)[0][0]
        return [zero_kx, zero_ky]

    def _lattice_grid_prepare(self,):
        len_nx, len_ny = self.nx, self.ny
        lattice_grid_2d = np.zeros((len_nx, len_ny, 2))
        half_nx, half_ny = len_nx//2, len_ny//2
        for i, j in product(range(-half_nx, half_nx+1), range(-half_ny, half_ny+1)):
            #* x-only displacement field from linear assumption
            lattice_grid_2d[i+half_nx,j+half_ny] = np.array([i*self.b_norm, j*self.b_norm])
            
        return lattice_grid_2d
    
    def _ft_main_fit(self, x, *coef, print_=False):
        #* x, y should be normalized by a / c
        x = x*2
        c = coef[0]
        n = (len(coef)-1)//self.n_coef
        
        y_ft = c
        prefac = 1/2
        for i in range(1, n+1):
            for j_ in range(1,self.n_coef//2+1):
                j = 2*j_
                y_ft += \
                coef[self.n_coef*i-j+1]*np.sin(prefac**j_*i*np.pi*x) \
                + coef[self.n_coef*i-j+2]*np.cos(prefac**j_*i*np.pi*x)
                
                if print_:
                    print(self.n_coef*i-j+1, self.n_coef*i-j+2, prefac**j_*i)
                
        return y_ft

    def optimize(self):
        
        neb_runs = {}
        key_list = []
        for key in self.element_info_buffer.keys():
            
            if key == self.specie_denote:
                continue

            # print(f'deriving local PN pot of {key} using NEB')

            ele_info = self.element_info_buffer[key]
            kernel_res = self.pn_kernel_buffer[key][self.deform_mode]
            u0 = ele_info['ux_opt']
            ufin = ele_info['ux_opt_mov']
            disso_x = ele_info['d_optimal']
            wx = ele_info['w_optimal']
            grid_x = ele_info['grid_x']
            
            u_buffer_init = image_linearinterp_uxfunc(u0, ufin, self.n_image, 
                                                    disso_x, wx, grid_x, 
                                                    self.b_norm)
            #! hyperparameter for neb
            n_neb_iter = 2500
            grad_coef = 1e-2
            u_buffer_optim, e_optim = neb_update(
                u_buffer_init, 
                n_iter=n_neb_iter,
                grad_coef=grad_coef, 
                zero_k_ind=self._zero_k_ind(),
                lattice_mesh=self._lattice_grid_prepare(),
                fmn_mesh=kernel_res,
                gsfe_coef=self.gsfe_coef, n_coef=self.gsfe_ncoef,
                gsfe_coef_grad=self.gsfe_grad_coef, n_coef_grad=self.gsfe_ncoef,
                b=self.b_norm,
                tol=1e-4, k_spring=0, #TODO tol and spring constant
                redist_freq=25,
                ) 

            neb_runs[key] = {
                'u_buffer_optim': u_buffer_optim,
                'e_optim': e_optim,
                'grid_x': grid_x,
            }
            
            key_list.append(key)
        
        print(f'NEB runs completed for {key_list}')
        return neb_runs
            
    def pn_extract(self,):
        neb_runsdict = self.optimize()
        
        local_pn_fit_buffer ={}
        for key in neb_runsdict.keys():
            
            pos_evo_list = []
            
            u_buffer_optim = neb_runsdict[key]['u_buffer_optim']
            e_optim = neb_runsdict[key]['e_optim']
            grid_x = neb_runsdict[key]['grid_x']
            #* record the position evolution 
            init_u = u_buffer_optim[0].flatten()
            grad_initu = np.gradient(init_u, edge_order=2)
            #TODO define the peak ind and track the position evolution
            peak_ind = find_peaks(grad_initu, height = 0.4)[0]
            peak_u_init = init_u[peak_ind[0]]

            # for i in range(n_image//2+1):
            for i in range(self.n_image):
                # plt.plot(grid_x, u_buffer_optim[i].flatten(), alpha=0.4, label=i)
                grad_u = np.gradient(u_buffer_optim[i].flatten(), edge_order=2)
                peak_u = u_buffer_optim[i][peak_ind[0]]+ u_buffer_optim[i][peak_ind[0]-1] + u_buffer_optim[i][peak_ind[0]+1]
                
                pos_evo_list.append(np.sum(init_u - u_buffer_optim[i].flatten()))
            
            # print(e_optim.flatten().shape, len(pos_evo_list))
            #* fit the gradient as local PN potential
            grad_tote = np.gradient(
                (e_optim.flatten()*self.b_norm),
                np.array(pos_evo_list),
                edge_order=2)/self.b_norm

            #TODO default setting of pn coef dimension
            n = 12 #* degree of polynomials
            self.n_coef = 4
            assert self.n_coef%2==0, print('Coefficient consists of sin-cos pair + 2 phase shift terms as Fourier basis')

            coef_init = np.ones((self.n_coef)*n+1)

            coef_grad_, _ = curve_fit(self._ft_main_fit, pos_evo_list/self.b_norm, grad_tote, p0=coef_init)
            local_pn_fit_buffer[key] = {
                'coef_grad_': coef_grad_,
                'n_coef': self.n_coef,
            }
            
        return local_pn_fit_buffer, self.n_coef