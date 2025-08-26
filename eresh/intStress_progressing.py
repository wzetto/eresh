import numpy as np
import pickle
import sympy as sp
import multiprocessing as mp    
from itertools import product, combinations
from scipy.fftpack import fft, ifft, fftfreq

#! under development 

def green_k(k_vec, c_ij, ijkl2ij_dict):
    k_norm = k_vec/np.linalg.norm(k_vec)
    zij = np.zeros((3,3))
    for (i, k) in product(range(3), range(3)):
        for (j, l) in product(range(3), range(3)):
            ji_denote = ijkl2ij_dict[(j,i)]
            kl_denote = ijkl2ij_dict[(k,l)]
            c_voigt = c_ij[ji_denote, kl_denote]
            zij[i,k] += c_voigt * k_norm[j] * k_norm[l]

    zij_inv = np.linalg.inv(zij).T
    return zij_inv/np.linalg.norm(k_vec)**2

def low_memo_ft(cart, freq, block_n=10):
    k_list = []
    freq_blocks = np.array_split(freq, block_n, axis=0)
    for k_block in freq_blocks:
        prod = np.sum(np.exp(-1j*np.dot(cart, k_block.T)), axis=0)
        k_list += prod.flatten().tolist()
    
    return np.array(k_list)

ijkl2ij_dict = {
    (0,0):0, (1,1): 1, (2,2): 2,
    (1,2): 3, (0,2): 4, (0,1): 5,
    (2,1): 3, (2,0): 4, (1,0): 5
}

#* some hyperparameters
calc_denote = '_superfast'
compo_space_denote = '_tivzrnbmohftawb2'
eva2gpa = 160.2176621 #* convert eV/Å^3 to GPa

#* specify the lattice
a, c = 3.075, 3.31 #TODO lattice constant
r_nx, r_ny, r_nz = 64, 64, 12 #* r-space resolution

ti, v, zr, nb, mo, hf, ta, w = 0.4, 0.25, 0.35, 0, 0, 0, 0, 0 #TODO composition

prim_frac = np.array([
    [0.5       , 0.        , 1/6],
    [0.        , 0.        , 2/3]
]) #* tm-frac coords

lattice_vec_prim = np.array([
    [a, 0, 0],
    [0,c, 0],
    [0, 0, a*np.sqrt(3)]
])
supercell_dim = np.array([r_nx, r_ny, r_nz])
super_frac = supercell_make(prim_frac, supercell_dim)
super_cart = np.dot(super_frac, lattice_vec_prim)

lattice_vol = np.linalg.det(lattice_vec_prim)
v_tmb2 = lattice_vol/2
lattice_vol_tot = lattice_vol*r_nx*r_ny*r_nz
n_cluster = r_nx*r_ny*r_nz*2

# * assign random spins to lattice sites
spin_type = [1,0,-1]
# spin_type = [1,-1]
#TODO generate random spin sequence
n_ti = np.random.randint(int(n_cluster*ti), int(n_cluster*ti+2))
n_v = np.random.randint(int(n_cluster*v), int(n_cluster*v+2))
n_zr = n_cluster - n_ti - n_v

# n_ti = np.random.randint(int(n_cluster*ti), int(n_cluster*ti+2))
# n_zr = np.random.randint(int(n_cluster*zr), int(n_cluster*zr+2))
# n_hf = n_cluster - n_ti - n_zr

# n_ti = np.random.randint(int(n_cluster*ti), int(n_cluster*ti+2))
# n_zr = np.random.randint(int(n_cluster*zr), int(n_cluster*zr+2))
# n_mo = n_cluster - n_ti - n_zr

# n_zr = np.random.randint(int(n_cluster*zr), int(n_cluster*zr+2))
# n_nb = np.random.randint(int(n_cluster*nb), int(n_cluster*nb+2))
# n_hf = n_cluster - n_zr - n_nb

# spin_num = [32768, 32768, 32768]
spin_num = [n_ti, n_v, n_zr]
# spin_num = [n_ti, n_zr, n_hf]
# spin_num = [n_ti, n_zr, n_mo]
# spin_num = [n_zr, n_nb, n_hf]

rand_seq = np.zeros(len(super_frac))
rand_seq[:spin_num[0]] = spin_type[0]
rand_seq[spin_num[0]:spin_num[1]+spin_num[0]] = spin_type[1]
rand_seq[spin_num[1]+spin_num[0]:] = spin_type[2]
np.random.shuffle(rand_seq)

ind_type1 = np.where(rand_seq == spin_type[0])[0]
ind_type2 = np.where(rand_seq == spin_type[1])[0]
ind_type3= np.where(rand_seq == spin_type[2])[0]
cart_type1 = super_cart[ind_type1]
cart_type2 = super_cart[ind_type2]
cart_type3 = super_cart[ind_type3]

#* sample the frequency grid (1120-0001-1100 lattice)
lx, ly, lz = a*r_nx, c*r_ny, np.sqrt(3)*a*r_nz 
#! doubling the k-resolution?
k_nx, k_ny, k_nz = r_nx*2, r_ny*2, r_nz*2  #* k-space resolution
kx = fftfreq(k_nx, lx/k_nx)*2*np.pi
ky = fftfreq(k_ny, ly/k_ny)*2*np.pi
kz = fftfreq(k_nz, lz/k_nz)*2*np.pi
kxyz = np.array(list(product(kx, ky, kz)))
#! remove zero-vector
kxyz = kxyz[1:]

#* calculate \theta_{alpha}(k)
#TODO artificial smearing coefficient
smear_coef1, smear_coef2, smear_coef3 = 1,1,1
theta_prefac1 = v_tmb2/lattice_vol_tot*np.exp(
    -1/2*(a**2*kxyz[:,0]**2*smear_coef1 
          + c**2*kxyz[:,1]**2*smear_coef1 
          + 3*a**2*kxyz[:,2]**2*smear_coef1))
theta_prefac2 = v_tmb2/lattice_vol_tot*np.exp(
    -1/2*(a**2*kxyz[:,0]**2*smear_coef2 
          + c**2*kxyz[:,1]**2*smear_coef2 
          + 3*a**2*kxyz[:,2]**2*smear_coef2))
theta_prefac3 = v_tmb2/lattice_vol_tot*np.exp(
    -1/2*(a**2*kxyz[:,0]**2*smear_coef3 
          + c**2*kxyz[:,1]**2*smear_coef3 
          + 3*a**2*kxyz[:,2]**2*smear_coef3))

#! memory sensitive
spin1_theta_k = theta_prefac1 * low_memo_ft(cart_type1, kxyz, block_n=16)
spin2_theta_k = theta_prefac2 * low_memo_ft(cart_type2, kxyz, block_n=16)
spin3_theta_k = theta_prefac3 * low_memo_ft(cart_type3, kxyz, block_n=16)
# spin1_theta_k = theta_prefac*np.sum(np.exp(-1j*np.dot(cart_type1, kxyz.T)), axis=0)
# spin2_theta_k = theta_prefac*np.sum(np.exp(-1j*np.dot(cart_type2, kxyz.T)), axis=0)
# spin3_theta_k = theta_prefac*np.sum(np.exp(-1j*np.dot(cart_type3, kxyz.T)), axis=0)

#* load stiffness tensor
#TODO elastic constant
# cij = np.load(f"dataset/cij/TiZrB_cij_{a}_{c}_0075250_try_2_fin.npy")
# cij = np.load(f"dataset/cij/VZrB_cij_{a}_{c}_0045550_try_1_fin.npy")
# cij = np.load(f"dataset/cij/TiVZrB2_cij_{a}_{c}_try_41_MACE.npy")
# cij = np.load(f"dataset/cij/TiZrHfB_cij_{a}_{c}_03333330_try_1.npy")
# cij = np.load(f"dataset/cij/TiZrMoB_cij_{a}_{c}_03333330_try_11.npy")
# cij = np.load(f"dataset/cij/TiVZrB_cij_{a}_{c}_05025250_try_37.npy")
# cij = np.load(f"dataset/cij/ZrVTiB_cij_{a}_{c}_05025250_try_23_DFT.npy")
# cij = np.load(f"dataset/cij/TiVZrB_cij_{a}_{c}_04030300_try_6_MACE.npy")
cij = np.load(f"dataset/cij/TiVZrB_cij_{a}_{c}_04025350_try_47_DFT.npy")
# cij = np.load(f"dataset/cij/ZrNbHfB_cij_{a}_{c}_03333330_try_1_fin.npy")
cij = cij/eva2gpa

#* derive eigenstrains
eigenstrain_mode_list = ['a', 'c', 'a_sqrt3']
lattice_axis_list = [1, 2, 3]
xti, xv, xzr, xnb, xmo, xhf, xta, xw = sp.symbols('TiB_2 VB_2 ZrB_2 NbB_2 MoB_2 HfB_2 TaB_2 WB_2')
spin1_eigenstraintensor = np.zeros((3,3))
spin2_eigenstraintensor = np.zeros((3,3))
spin3_eigenstraintensor = np.zeros((3,3))

for eigenstrain_mode, lattice_axis in zip(eigenstrain_mode_list, lattice_axis_list):
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
    
    #TODO specify eigenstrain
    spin1_eigenstraintensor[lattice_axis-1, lattice_axis-1] = ti_eigen
    spin2_eigenstraintensor[lattice_axis-1, lattice_axis-1] = v_eigen
    spin3_eigenstraintensor[lattice_axis-1, lattice_axis-1] = zr_eigen
 
    # spin1_eigenstraintensor[lattice_axis-1, lattice_axis-1] = ti_eigen
    # spin2_eigenstraintensor[lattice_axis-1, lattice_axis-1] = zr_eigen
    # spin3_eigenstraintensor[lattice_axis-1, lattice_axis-1] = hf_eigen 
    
    # spin1_eigenstraintensor[lattice_axis-1, lattice_axis-1] = ti_eigen
    # spin2_eigenstraintensor[lattice_axis-1, lattice_axis-1] = zr_eigen
    # spin3_eigenstraintensor[lattice_axis-1, lattice_axis-1] = mo_eigen 

    # spin1_eigenstraintensor[lattice_axis-1, lattice_axis-1] = zr_eigen
    # spin2_eigenstraintensor[lattice_axis-1, lattice_axis-1] = nb_eigen
    # spin3_eigenstraintensor[lattice_axis-1, lattice_axis-1] = hf_eigen
    
#* derive green's function
green_k_list = []
for k in kxyz:
    green_k_list.append(green_k(k, cij, ijkl2ij_dict))
    
def sigma_k_main(k_i):
    k_vec = kxyz[k_i]
    sigma_k = np.zeros((3,3), dtype=complex)
    green_k = green_k_list[k_i]
    
    theta_k1 = spin1_theta_k[k_i]
    theta_k2 = spin2_theta_k[k_i]
    theta_k3 = spin3_theta_k[k_i]
    
    for i, j in product(range(3), repeat=2):
        for k, l, p, q, l, m in product(range(3), repeat=6):
            sigma_k[i,j] += \
                cij[ijkl2ij_dict[(i,j)], ijkl2ij_dict[(k,l)]] * \
                green_k[k,p] * \
                k_vec[q] * k_vec[l] * \
                cij[ijkl2ij_dict[(p,q)], ijkl2ij_dict[(m,m)]] * \
                (spin1_eigenstraintensor[m,m] * theta_k1
                + spin2_eigenstraintensor[m,m] * theta_k2
                + spin3_eigenstraintensor[m,m] * theta_k3
                )
                
        for m in range(3):
            sigma_k[i,j] -= \
                cij[ijkl2ij_dict[(i,j)], ijkl2ij_dict[(m,m)]] * \
                (spin1_eigenstraintensor[m,m] * theta_k1 
                  + spin2_eigenstraintensor[m,m] * theta_k2
                  + spin3_eigenstraintensor[m,m] * theta_k3
                 )
                
    return sigma_k

pool = mp.Pool(22)
sigma_k_list = pool.map(sigma_k_main, range(len(kxyz)))       


sigma_k_list_forloop = []

for k_i, k_vec in enumerate(kxyz):
    
    if k_i > 1000:
        break #* debug
    
    sigma_k = np.zeros((3,3), dtype=complex)
    green_k = green_k_list[k_i]
    
    theta_k1 = spin1_theta_k[k_i]
    theta_k2 = spin2_theta_k[k_i]
    theta_k3 = spin3_theta_k[k_i]
    
    for i, j in product(range(3), repeat=2):
        for k, l, p, q, l, m in product(range(3), repeat=6):
            sigma_k[i,j] += \
                cij[ijkl2ij_dict[(i,j)], ijkl2ij_dict[(k,l)]] * \
                green_k[k,p] * \
                k_vec[q] * k_vec[l] * \
                cij[ijkl2ij_dict[(p,q)], ijkl2ij_dict[(m,m)]] * \
                (spin1_eigenstraintensor[m,m] * theta_k1
                + spin2_eigenstraintensor[m,m] * theta_k2
                + spin3_eigenstraintensor[m,m] * theta_k3
                )
                
        for m in range(3):
            sigma_k[i,j] -= \
                cij[ijkl2ij_dict[(i,j)], ijkl2ij_dict[(m,m)]] * \
                (spin1_eigenstraintensor[m,m] * theta_k1 
                  + spin2_eigenstraintensor[m,m] * theta_k2
                  + spin3_eigenstraintensor[m,m] * theta_k3
                 )
                
    sigma_k_list_forloop.append(sigma_k)
    
def low_memo_mat_prod(a, b, prefac, seperate_block_n = 10):
    sum_a_dim_b = []
    b_blocks = np.array_split(b, seperate_block_n, axis=0) #* split b into blocks to save memory
    for b_block_i in b_blocks:
        prod = np.exp(1j*(a@b_block_i.T)) #* k x r_block
        sum_a_dim_b += (np.sum(prefac*prod, axis=0).flatten()).tolist() #* sum over k
    return np.array(sum_a_dim_b)

#TODO fine r-grid sampling to be used in pf model
r_nx_sampling, r_ny_sampling, r_nz_sampling = 128, 128, 1  

#* sampling grid preparation in Cartesian coordinates
prim_frac_grid = np.array([0.       , 0.        , 2/3]).reshape(1,-1)
#TODO grid lattice vectors
lattice_vec_grid = np.array([
                        [a/2, 0, 0],
                        [0,c/2, 0],
                        [0, 0, a*np.sqrt(3)]
                    ])
sample_grid_dim = np.array([r_nx_sampling, r_ny_sampling, r_nz_sampling])
super_frac_grid = supercell_make(prim_frac_grid, sample_grid_dim)
super_cart_grid = np.dot(super_frac_grid, lattice_vec_grid)

#TODO use grid coordinates to calculate sigma_r
sigma_r_list = np.zeros((len(super_cart_grid), 3, 3), dtype=float)
# ft_prefac = np.sum(np.exp(1j*np.dot(kxyz, super_cart.T)), axis=0)
# ft_prefac = low_memo_mat_prod(kxyz, super_cart.T)
for i, j in combinations(range(3), 2):
    sigma_ij_k_list = np.array([sigma_k[i,j] for sigma_k in sigma_k_list])
    # sigma_ij_r_list = sigma_ij_k_list.reshape(-1,1)*ft_prefac
    sigma_ij_r_list = low_memo_mat_prod(kxyz, super_cart_grid, sigma_ij_k_list.reshape(-1,1),
                                        seperate_block_n = 12)

    sigma_r_list[:,i,j] = np.real(sigma_ij_r_list.flatten())
    if i != j:
        sigma_r_list[:,j,i] = np.real(sigma_ij_r_list.flatten())
        
lattice_vec_prim = np.array([
    [a, 0, 0],
    [0,c, 0],
    [0, 0, a*np.sqrt(3)]
])

#* 2110-0001-0110 1100 coordinate system
#! verify the order of vector!
nmt_raw = (np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1],
])@lattice_vec_prim)
nmt_raw = nmt_raw/np.linalg.norm(nmt_raw, axis=1)[:,None]

#* 1100 - -1100 - 0001 coordinate system
#! verify the order of vector!
nmt_n = np.array([
    [a/np.sqrt(a**2+c**2), c/np.sqrt(a**2+c**2), 0],
    [-c/np.sqrt(a**2+c**2), a/np.sqrt(a**2+c**2), 0],
    [0,0,1],
])

rot_tensor = nmt_n@np.linalg.inv(nmt_raw)
print('Orthogonality check:\n', np.round(rot_tensor@rot_tensor.T, 2))

sigma_r_list_rot = np.zeros_like(sigma_r_list)
for sigma_i in range(sigma_r_list.shape[0]):
    
    #* ein-sum method
    # sigma_r_rot = np.zeros((3,3))
    # sigma_r = sigma_r_list[sigma_i]
    # for i, j in product(range(3), repeat=2):
    #     for m, n in product(range(3), repeat=2):
    #         sigma_r_rot[i,j] += rot_tensor[i,m]*rot_tensor[j,n]*sigma_r[m,n]
    
    #* mat-mul method
    sigma_r_rot = np.dot(rot_tensor, np.dot(sigma_r_list[sigma_i], rot_tensor.T))
    
    sigma_r_list_rot[sigma_i] = sigma_r_rot
    
z_specify = np.unique(super_cart_grid[:,2])
cart_ind = np.where(np.abs(super_cart_grid[:,2] - z_specify) < 1e-3)[0]

sigma_13_list, sigma_23_list = sigma_r_list_rot[:,0,2], sigma_r_list_rot[:,1,2]

sigma_13_specify = sigma_13_list[cart_ind]
sigma_23_specify = sigma_23_list[cart_ind]  
cart_specify = super_cart_grid[cart_ind]

stress_field_dict = {
    'coord_grid': super_cart_grid,
    'sigma_13': sigma_13_list,
    'sigma_23': sigma_23_list,
    'grid_dimension': sample_grid_dim,
    'lattice_dimension': supercell_dim,
    'coord_atom': super_cart,
    'atom_spin': rand_seq,
    'lattice_param': [a, c],
}

compo_denote = '40Ti25V35ZrB2'
create_dir(sav_pth+f'/{compo_denote}')
with open(f'{sav_pth}/{compo_denote}/stress_{r_nx_sampling}_{r_ny_sampling}_{r_nz_sampling}.pkl', 'wb') as f:
    pickle.dump(stress_field_dict, f)
    
print(f'Stress field saved to {compo_denote}/stress_{r_nx_sampling}_{r_ny_sampling}_{r_nz_sampling}.pkl')