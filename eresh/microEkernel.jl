module microEkernel

    using LinearAlgebra
    using IterTools
    using NPZ
    using FFTW
    using Base.Threads
    println("Currently using $(Threads.nthreads()) threads.")

    export lattice2d, pn_kernel

    struct lattice2d 
        specie_denote::String
        stiffness_tensor_buffer::Dict{String, Matrix{Float64}}
        eva2gpa::Float64
        mjm2eva::Float64
        voigt_dict::Dict
        permu_dict::Dict
        pn_pot_grid::Vector{Int64}
        lattice_param::Vector{Float64}
        b_norm::Float64
        pn_kernel_savpth::String
    end

    function adjoint_mat(x)
        return det(x)*inv(x)
    end

    function d_kxkykz(cij, k_vec, ijkl2ij_dict)
        d_mat = Complex.(zeros(3,3))
        for (i,j,m,n) in product(1:3, 1:3, 1:3, 1:3)
            im_denote, jn_denote = ijkl2ij_dict[(i,m)], ijkl2ij_dict[(j,n)]
            km, kn = k_vec[m], k_vec[n]
            d_mat[i,j] += cij[im_denote,jn_denote]*km*kn
        end
        return d_mat
    end

    function permu_sign(permu, permu_dict)
        if permu in keys(permu_dict)
            return permu_dict[permu]
        else
            return 0
        end
    end 

    function f_mn(kx, ky, cij, k3_list, ijkl2ij_dict, permu_dict)
        sum_f_13, sum_f_23, sum_g_13, sum_g_23 = 0, 0, 0, 0
        dk3 = diff(k3_list)[1]
        n = 3 #* for shear components
        for k3 in k3_list
            if norm([kx, ky, k3]) < 1e-6
                continue
            end

            k_vec = [kx, ky, k3]
            d_mat = d_kxkykz(cij, k_vec, ijkl2ij_dict)
            n_mat = adjoint_mat(d_mat)
            d_det = det(d_mat)

            for (j,l,p,q,s) in product(1:3, 1:3, 1:3, 1:3, 1:3)

                mn_denote_1n, mn_denote_2n = ijkl2ij_dict[(1,n)], ijkl2ij_dict[(2,n)]
                jl_denote = ijkl2ij_dict[(j,l)]
                pq_denote = ijkl2ij_dict[(p,q)]
                f_1s_denote, g_2s_denote = ijkl2ij_dict[(1,s)], ijkl2ij_dict[(2,s)]

                cmnjl_1n, cmnjl_2n = cij[mn_denote_1n,jl_denote], cij[mn_denote_2n,jl_denote]
                cpq1s_f, cpq2s_g = cij[pq_denote,f_1s_denote], cij[pq_denote,g_2s_denote]
                
                kq = k_vec[q]
                n_lp = n_mat[l,p]
                e_js2 = permu_sign((j,s,2), permu_dict)
                e_js1 = permu_sign((j,s,1), permu_dict)

                sum_f_13 += cmnjl_1n*cpq1s_f*kq*n_lp/(2*pi*d_det)*(kx*e_js2-ky*e_js1)*dk3
                sum_f_23 += cmnjl_2n*cpq1s_f*kq*n_lp/(2*pi*d_det)*(kx*e_js2-ky*e_js1)*dk3
                sum_g_13 += cmnjl_1n*cpq2s_g*kq*n_lp/(2*pi*d_det)*(kx*e_js2-ky*e_js1)*dk3
                sum_g_23 += cmnjl_2n*cpq2s_g*kq*n_lp/(2*pi*d_det)*(kx*e_js2-ky*e_js1)*dk3
            end
        end
        return sum_f_13, sum_f_23, sum_g_13, sum_g_23
    end

    function pn_kernel(params::lattice2d)

        b = params.b_norm
        #* initialize frequency grid
        len_nx, len_ny = params.pn_pot_grid[1], params.pn_pot_grid[2]
        len_kx, len_ky = len_nx*2, len_ny #* for mirroring the grid
        lx, ly = b*len_nx, b*len_ny #! grid spacing = atomic spacing in PN pot derivation
        kx, ky = fftfreq(len_kx, len_nx/lx)*2*pi, fftfreq(len_ky, len_ny/ly)*2*pi

        stiffness_buffer = params.stiffness_tensor_buffer
        kernel_dict = Dict{String, Any}()
        for specie_name in keys(stiffness_buffer)
            cij = stiffness_buffer[specie_name]/params.eva2gpa #* convert to eV/Å^3
            #* fine grid along k3
            kz_lim = 50
            kz_sample = 8001
            kz = range(-kz_lim, kz_lim, length=kz_sample)
            kz = Complex.(kz)
            f_13_mesh = Complex.(zeros((len_kx, len_ky)))
            f_23_mesh = Complex.(zeros((len_kx, len_ky)))
            g_13_mesh = Complex.(zeros((len_kx, len_ky)))
            g_23_mesh = Complex.(zeros((len_kx, len_ky)))

            prod_kxky = collect(product(1:len_kx, 1:len_ky))
            Threads.@threads for (ki, kj) in prod_kxky
                kx_val, ky_val = kx[ki], ky[kj]
                f_13, f_23, g_13, g_23 = f_mn(kx_val, ky_val, cij, kz, params.voigt_dict, params.permu_dict)
                f_13_mesh[ki, kj] = f_13
                f_23_mesh[ki, kj] = f_23
                g_13_mesh[ki, kj] = g_13
                g_23_mesh[ki, kj] = g_23
            end

            kernel_dict[specie_name] = Dict(
                "f_13"=>f_13_mesh,
                "f_23"=>f_23_mesh,
                "g_13"=>g_13_mesh,
                "g_23"=>g_23_mesh,
            )
            
            npzwrite(params.pn_kernel_savpth*"/$(specie_name)_f13.npy", f_13_mesh)
            npzwrite(params.pn_kernel_savpth*"/$(specie_name)_f23.npy", f_23_mesh)
            npzwrite(params.pn_kernel_savpth*"/$(specie_name)_g13.npy", g_13_mesh)
            npzwrite(params.pn_kernel_savpth*"/$(specie_name)_g23.npy", g_23_mesh)
            println("PN kernel for $(specie_name) calculated")
        end 

        
        # println("PN kernel saved to $(params.pn_kernel_savpth)/$(params.specie_denote).npz")
        return kernel_dict
    end
end