import numpy as np
from numpy.lib.npyio import NpzFile

from scipy.interpolate import RectBivariateSpline
import pickle
import copy
from typing import List

import os

from eresh import utils
from eresh import lineE
from eresh import pfdd
from eresh import neb
from eresh.constants import VOIGT_DICT, MJsM2EVA, EVcA2GPA, VOIGT_DICT_jl, PERMU_DICT_jl

#* main function for evolution of profile
class disloc_profile:
    def __init__(self,
                
                specie_denote: str = None,   
                lattice_info : NpzFile = None,
                stiffness_tensor_buffer: NpzFile = None,
                constituent_info : NpzFile = None,
                
                gsfe_func = None,
                gsfe_coef_buffer: NpzFile = None,
                
                use_internal_stress : bool = False,
                int_stress_grid : List[int] = [128, 128],
                fine_grid_int_stress : int = 2,
                internal_stress_buffer: NpzFile = None,
                
                disloc_type: str = 'edge',
                outer_cut : float = 1000,
                lt_generator = None,
                
                pn_calc: bool = True,
                pn_pot_grid : List[int] = [513, 1],
                pn_kernel_buffer : NpzFile = None,
                pn_weight_buffer : NpzFile = None,
                pn_savpth : str = None,
                
                neb_image : int = 51,
                
                external_load_list : List[float] = None,
                
                evo_grid : List[int] = [128, 128],
                fine_grid_evo : int = 2,
                
                evo_sav_path: str = None,
                threads_num: int = 1,
                verbose: bool = False,):

        #* compositional information
        self.specie_denote = specie_denote
        self.constituent_info = constituent_info
        
        #* structural information
        self.lattice_param = lattice_info['lattice_param']
        self.a, self.c = self.lattice_param
        self.lattice_vec_origin = lattice_info['lattice_vec']
        self.lattice_vec_prismac = np.array([
            [self.a/np.sqrt(self.a**2+self.c**2), self.c/np.sqrt(self.a**2+self.c**2), 0],
            [-self.c/np.sqrt(self.a**2+self.c**2), self.a/np.sqrt(self.a**2+self.c**2), 0],
            [0,0,1],
        ]) #* 1123-1123-1100 lattice vector

        #* constants
        self.voigt_dict = VOIGT_DICT  
        self.permu_dict_jl = PERMU_DICT_jl
        self.voigt_dict_jl = VOIGT_DICT_jl
        self.mjm2eva = MJsM2EVA 
        self.eva2gpa = EVcA2GPA 
        self.stiffness_tensor_buffer = stiffness_tensor_buffer

        #* stacking fault energy as function of shear displacement on {1100} plane
        self.sfe_func = gsfe_func or utils.ft_main_vec
        self.sfe_coef = gsfe_coef_buffer['gsfe_coef']
        self.sfe_grad_coef = gsfe_coef_buffer['gsfe_grad_coef']
        self.sfe_n_coef = gsfe_coef_buffer['n_coef']
        self.isfe = (self.sfe_func(np.array([1/2]), *self.sfe_coef, n_coef = self.sfe_n_coef)*self.mjm2eva)[0] #* in eV/A^2
        
        #* disloction structural information
        self.b_partial_vec = np.array([self.a/2, self.c/2, 0])
        self.b_norm = np.sqrt(self.a**2 + self.c**2)
        self.lt_prelog = np.log(outer_cut/self.b_norm) #* prelog factor for line tension
        self.disloc_type = disloc_type
        self.lt_generator = lt_generator
        
        #* pn potential information
        self.pn_calc = pn_calc
        self.pn_pot_grid = np.array(pn_pot_grid)
        self.pn_savpth = pn_savpth
        self.pn_kernel_buffer = pn_kernel_buffer
        self.pn_weight_buffer = pn_weight_buffer
        
        #* neb hyperparameter
        self.neb_image = neb_image
        
        #* internal stress information
        self.use_internal_stress = use_internal_stress
        self.taup_grid = np.array(int_stress_grid)
        if disloc_type == 'edge':
            self.int_mat = internal_stress_buffer['sigma_13']
        elif disloc_type == 'screw':
            self.int_mat = internal_stress_buffer['sigma_23']
            
        #* external loading (shear stress)
        self.external_load_list = external_load_list
        
        #* evolution setup 
        self.evo_grid = np.array(evo_grid)
        self.fine_grid_evo = fine_grid_evo
        
        #* configuration
        self.sav_path = evo_sav_path
        self.threads_num = threads_num    
        self.verbose = verbose
        
    def _pn_kernel(self,):
        
        print('PN kernel not provided, calculating from scratch')
        #* prepare rot cij buffer
        stiffness_tensor_buffer_1123 = {}
        for key, cij_raw in self.stiffness_tensor_buffer.items():
            stiffness_tensor_buffer_1123[key] = self._cij_rot(cij_raw)

        os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
        os.environ["JULIA_NUM_THREADS"] = f"{self.threads_num}"
        from juliacall import Main as jl
        
        jl.seval('include("eresh/microEkernel.jl"); using .microEkernel')
    
        pnkernel_param = jl.microEkernel.lattice2d(
                            self.specie_denote,
                            stiffness_tensor_buffer_1123,
                            self.eva2gpa,
                            self.mjm2eva,
                            self.voigt_dict_jl,
                            self.permu_dict_jl,
                            self.pn_pot_grid,
                            self.lattice_param,
                            self.b_norm,
                            self.pn_savpth + '/kernel'
                        )

        kernel_dict = jl.microEkernel.pn_kernel(pnkernel_param)
        
        #* convert to python dict
        kernel_dict_py = {}
        for key in stiffness_tensor_buffer_1123.keys():
            
            f13 = np.load(self.pn_savpth + f'/kernel/$(key)_f13.npy')
            f23 = np.load(self.pn_savpth + f'/kernel/$(key)_f23.npy')
            g13 = np.load(self.pn_savpth + f'/kernel/$(key)_g13.npy')
            g23 = np.load(self.pn_savpth + f'/kernel/$(key)_g23.npy')
            
            kernel_dict_py[key] = {
                'f13': f13,
                'f23': f23,
                'g13': g13,
                'g23': g23,
            }
        
        return kernel_dict
       
    def _cij_rot(self, cij):
        lvec_raw = self.lattice_vec_origin/np.linalg.norm(self.lattice_vec_origin, axis=1)[:,None]
        lvec_pac = self.lattice_vec_prismac.copy()
        
        sijkl_raw = utils.ijkl_invert(np.linalg.inv(cij), self.voigt_dict)
        cij_rot = utils.cij_rot(sijkl_raw, lvec_raw, lvec_pac, self.voigt_dict)
        return cij_rot 
    
    def _lt_prefac(self, cij):
        _lt = lineE.lt_prefactor(cij, self.b_norm, self.b_partial_vec,)
        return _lt()

    def _intstress_interp(self, rand_tau_mat):
        #* random shear stress field
        if self.use_internal_stress:
            #* interpolation 
            rand_tau_interp = RectBivariateSpline(
                np.arange(rand_tau_mat.shape[0]), 
                np.arange(rand_tau_mat.shape[1]), 
                rand_tau_mat, 
            )
        else:
            rand_tau_interp = None
        
        return rand_tau_interp

    def _pn_info_derive(self,):
        
        ''' 
        Optimal dissociation + core width from 1D PFDD
        '''
        
        pn_kernel_buffer = self.pn_kernel_buffer or self._pn_kernel()
        pfdd_param = pfdd.pfdd(
            self.mjm2eva, self.eva2gpa, self.disloc_type, self.specie_denote,
            self.pn_pot_grid[0], self.pn_pot_grid[1], self.lattice_param,
            self.b_norm, self.stiffness_tensor_buffer,
            self.sfe_coef, self.sfe_grad_coef, self.sfe_n_coef,
            pn_kernel_buffer, self.threads_num
        )
        
        return pfdd_param.heb_ele_optimal()

    def _local_pn_coef(self):
        
        pn_kernel_buffer = self.pn_kernel_buffer or self._pn_kernel()
        element_info_buffer, heb_opt_d_w = self._pn_info_derive()
        neb_param = neb.neb(
            self.mjm2eva, self.eva2gpa, self.disloc_type, self.specie_denote,
            self.pn_pot_grid[0], self.pn_pot_grid[1], self.lattice_param,
            self.b_norm, self.stiffness_tensor_buffer,
            self.sfe_coef, self.sfe_grad_coef, self.sfe_n_coef,
            pn_kernel_buffer, self.threads_num,
            element_info_buffer, self.neb_image,
        )

        return neb_param.pn_extract(), heb_opt_d_w
    
    def _pn_coef_weight_buffer(self, pn_coef_eleinfo):
        
        pn_coef_buffer = np.sum(
            np.array([self.pn_weight_buffer[key][...,None]*pn_coef_eleinfo[key]['coef_grad_']
                      for key in self.pn_weight_buffer.keys()]), axis=0
        )

        return pn_coef_buffer
        
    def evo_jac(self, y1, y2, d_init, #* 1 x nx vectors
            lt_prefac,
            tau_ext,
            tau_p_interp, taup_xdim, taup_ydim, #* random shear stress field
            pn_coef_buffer, n_coef, #* random pn potential part
            fine_grid_n, 
            mode = None,
            pn_calc = True, #* if incorporate pn potential
        ):
    
        ''' 
        y2 lead
        y1 trail
        '''

        b = self.b_norm
        nx, ny = self.evo_grid
        
        #* average partial part 
        y1_ave, y2_ave = np.mean(y1), np.mean(y2)
        dy1 = np.abs(y2_ave - y1) * b / fine_grid_n
        dy2 = np.abs(y2 - y1_ave) * b / fine_grid_n
        
        #TODO type of partial disloc interaction force
        fy1 = -1/2*self.isfe/(b/2)*(d_init/dy1 - 1) #* force on y1
        fy2 = 1/2*self.isfe/(b/2)*(d_init/dy2 - 1) #* force on y2
        
        #* lt part 
        y1_smooth, y2_smooth = copy.deepcopy(y1), copy.deepcopy(y2)
        d2_y1 = utils.second_derive(y1_smooth*b/fine_grid_n, b/fine_grid_n)
        d2_y2 = utils.second_derive(y2_smooth*b/fine_grid_n, b/fine_grid_n)
        d2_y1_grad = d2_y1*lt_prefac/b
        d2_y2_grad = d2_y2*lt_prefac/b
        
        #* tau_p part
        if tau_p_interp is not None:
            taup_y1 = tau_p_interp.ev(np.arange(nx) % taup_xdim, y1 % taup_ydim)/2
            taup_y2 = tau_p_interp.ev(np.arange(nx) % taup_xdim, y2 % taup_ydim)/2
        else:
            taup_y1 = 0
            taup_y2 = 0

        #* peierls potential part
        if pn_calc:
            
            y2_pnloc = np.floor(y2).astype(int) % taup_ydim #* periodic normalized
            y1_pnloc = np.floor(y1).astype(int) % taup_ydim
            x_loc = np.arange(nx) % taup_xdim
            
            #* moving distance from ideal disloc pos
            dy2_pn = (y2*2 - np.floor(y2*2)) #* periodic normalized
            dy1_pn = (y1*2 - np.floor(y1*2))
            
            pn_y2 = []
            pn_y1 = []
            
            #* N_y * M_coef dim
            coef_buffer_y2 = np.array([pn_coef_buffer[x_loc_i, y_loc_i] 
                            for x_loc_i, y_loc_i in zip(x_loc, y2_pnloc)])
            coef_buffer_y1 = np.array([pn_coef_buffer[x_loc_i, y_loc_i]
                            for x_loc_i, y_loc_i in zip(x_loc, y1_pnloc)])
            
            pn_y2 = -utils.ft_main_vec(dy2_pn, *coef_buffer_y2, 
                                n_coef=n_coef, b=b)
            pn_y1 = -utils.ft_main_vec(dy1_pn, *coef_buffer_y1,
                                n_coef=n_coef, b=b)
                
            pn_y2 = pn_y2.flatten()
            pn_y1 = pn_y1.flatten()
            
        else:
            pn_y1 = np.zeros_like(y1)
            pn_y2 = np.zeros_like(y2)
        
        grad_y1 = fy1 + tau_ext/2 + taup_y1 + d2_y1_grad + pn_y1
        grad_y2 = fy2 + tau_ext/2 + taup_y2 + d2_y2_grad + pn_y2
        
        if mode == 'print':
            return [fy1, fy2], [taup_y1, taup_y2], [d2_y1_grad, d2_y2_grad], [pn_y1, pn_y2]
        
        return grad_y1, grad_y2

    def evo_gl_minimize(self, y1, y2, d_init, lt_prefac, tau_ext,
                    tau_p_interp, taup_xdim, taup_ydim,
                    pn_coef_buffer, n_coef_buffer, pn_calc,
                    fine_grid_n, 
                    m, #* drag coef
                    max_iter = 1000, tol = 1e-6,
                    ):
        
        b = self.b_norm
        max_iter_reach = False
        ave_pos_list = []
        for i in range(max_iter):
            grad_y1, grad_y2 = self.evo_jac(
                    y1, y2, d_init, lt_prefac, 
                    tau_ext, tau_p_interp, taup_xdim, taup_ydim,
                    pn_coef_buffer, n_coef_buffer,
                    fine_grid_n, pn_calc=pn_calc
                )
            # print(grad_y1)
            y1 += grad_y1*m
            y2 += grad_y2*m
            
            norm_grady1 = np.linalg.norm(grad_y1)
            norm_grady2 = np.linalg.norm(grad_y2)
            
            if len(ave_pos_list) > 10000:
                #! metric for static configuration
                pos_list_past = ave_pos_list[-20000:]
                k_fitavepos, _ = np.polyfit(np.arange(len(pos_list_past)), pos_list_past, 1)
                if ((norm_grady1 < tol and norm_grady2 < tol)
                    or (k_fitavepos < 1e-6)):
                    if self.verbose:
                        print(f'Converged at iteration {i} \
                            max norm grad = {np.max([norm_grady1, norm_grady2])}, \
                                k_fitavepos = {k_fitavepos}')
                    break
            else:
                if np.linalg.norm(grad_y1) < tol and np.linalg.norm(grad_y2) < tol:
                    if self.verbose:
                        print(f'Converged at iteration {i} \
                            max norm grad = {np.max([norm_grady1, norm_grady2])}')
                    break
            
            if i == max_iter - 1:
                max_iter_reach = True
                print(f'tau = {np.round(tau_ext*self.eva2gpa,3)} GPa; Max iterations reached: {max_iter}, \
                    grad norm = {np.linalg.norm(grad_y1)} {np.linalg.norm(grad_y2)} \
                    k_fitavepos = {k_fitavepos}')
            
            ave_pos_list.append((np.mean(y1) + np.mean(y2))/2*b/fine_grid_n)
            
        y_ave_pos = (np.mean(y1) + np.mean(y2))/2*b/fine_grid_n
        ave_dissodis = (np.mean(y2) - np.mean(y1))*b/fine_grid_n
        if self.verbose:
            print(f'tau = {np.round(tau_ext*self.eva2gpa,3)} GPa; ave pos {round(y_ave_pos, 2)} A; ave disso distance {round(ave_dissodis, 2)} A')
        
        #* record the force
        # grad_info_fin = jac(
        #     y1, y2, d_init, b, k, lt_prefac, 
        #     tau_ext, tau_p_interp, taup_xdim, taup_ydim,
        #     pn_coef_buffer, n_coef_buffer,
        #     fine_grid_n,
        #     mode='print',
        #     pn_calc=pn_calc,
        #     )
        
        return y1, y2, max_iter_reach, [grad_y1, grad_y2]
    
    def evolute(self,):
        
        print(f'Stacking fault energy: {self.isfe} eV/A^2')
        print('==============================')
        print(f'Start line energy prefactor calculation for {self.specie_denote} {self.disloc_type} dislocation')
        
        #* lt prefactor
        cij_homogeneous = self._cij_rot(self.stiffness_tensor_buffer[self.specie_denote])/self.eva2gpa
        ''' 
        - 1. Currently only support homogeneous line tension, may upgrade to localized form in the future
        - 2. Also, due to limitation of classical theory an inner cutoff is attached to path integral. May upgrade to non-singular form in the future
        - 3. lt_generator is a function of line angle, currently only support pi/2 throughout the evolution. May upgrade to a function of line angle in the future.
        '''
        lt_generator = self.lt_generator or self._lt_prefac(cij_homogeneous)
        if self.lt_generator is not None:
            print(f'Loading line energy prefactor from provided function')
            
        if self.disloc_type == 'edge':
            lt = lt_generator(np.pi/2)*self.lt_prelog
        elif self.disloc_type == 'screw':
            lt = lt_generator(0)* self.lt_prelog
            
        print(f'Line energy prefactor: {lt} eV/A')
        print('==============================')
        
        #* internal stress parameterization
        #! derivation program is progressing, currently provide some calculated stress field
        print(f'Loading internal stress field by default')
        print('==============================')
        taup_xdim, taup_ydim = self.taup_grid
        rand_tau_interp = self._intstress_interp(self.int_mat)
        
        #* local pn parameterization
        print(f'Start local PN potential parameterization')
        (localpn_coef_buffer, n_coef_pn), heb_opt_d_w = self._local_pn_coef()
        pn_coef_buffer = self._pn_coef_weight_buffer(localpn_coef_buffer)
        d_init = heb_opt_d_w[0]
        
        #* create inital disloc profile
        trail_y = np.zeros(self.evo_grid[0])
        lead_y = trail_y + d_init/(self.b_norm/self.fine_grid_evo)
        
        #* start evolution
        print('==============================')
        if self.external_load_list is None:
            return print('No external load provided, exiting')
        
        print('Start evolution')
        runs_dict = {}
        
        for tau in self.external_load_list:
            
            trail_y, lead_y, max_iter_reach, grad_list = self.evo_gl_minimize(
                trail_y, lead_y, d_init, lt,
                tau, rand_tau_interp, taup_xdim, taup_ydim,
                pn_coef_buffer, n_coef_pn, self.pn_calc,
                self.fine_grid_evo,
                m=1e-1, max_iter=150000, tol=1e-6,
            )
            # [fy1_iter, fy2_iter], [taup_y1_iter, taup_y2_iter], [d2_y1_grad_iter, d2_y2_grad_iter], [pn_y1_iter, pn_y2_iter] = grad_info
            
            runs_dict[round(tau*self.eva2gpa, 3)] = {
                'y1': trail_y % self.evo_grid[1],
                'y2': lead_y % self.evo_grid[1],
                'disloc_pos': (np.mean(trail_y) + np.mean(lead_y))/2*self.b_norm/self.fine_grid_evo,
                'grad_list': grad_list,
                # 'fy_partialint': [fy1_iter, fy2_iter],
                # 'taup': [taup_y1_iter, taup_y2_iter],
                # 'lt': [d2_y1_grad_iter, d2_y2_grad_iter],
                # 'pn': [pn_y1_iter, pn_y2_iter],
            }

            #* save the runs_dict
            with open(f'{self.sav_path}/evo_runs/{self.specie_denote}{self.disloc_type}.pickle', 'wb') as f:
                pickle.dump(runs_dict, f)
                
            if max_iter_reach:
                break
        
        print(f'Critical stress reached, information saved to {self.sav_path}/evo_runs/{self.specie_denote}{self.disloc_type}.pickle')
        
        return