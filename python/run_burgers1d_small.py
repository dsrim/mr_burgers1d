
from burgers1d import Rom
import numpy as np
import matplotlib.pyplot as pl
from copy import copy
import os,sys


Rom0 = Rom()

if 0:
    Rom0.sample_hfm(N=250,dt=0.0125,xl=0.,xr=100.,M=4000)
    #Rom0.sample_hfm(N=500,dt=0.006125,xl=0.,xr=100.,M=8000)
    
    Rom0.plot_hfm_solution()
    Rom0.build_bases(tol=1e-10,max_basis_size=200,Mfinal=1000,t_interval=20,\
                   nalpha=4,P=4)
    Rom0.save_data()

Rom0.load_data()
#Rom0.mc_sample()
#Rom0._reduce_bases(tol=1e-6)
Rom0._rom_reduced=True

mu_list = []
rs_list = []
for k in range(100000):
    mu0 = Rom0._random_mu()
    Rom0.run_rom(mu=mu0)
    print('sample no. = ' + str(k))
    mu_list.append(copy(mu0))
    rs_list.append(copy(Rom0._rom_r_list))

    if np.mod(k+1 , 2000) == 0:
        sol_fname = 'sampled_rom_sols_' + str(k) + '.npy'
        mu_fname = 'sampled_mu_' + str(k) + '.npy'
        print('- saving to file: ' + sol_fname)
        print('- saving to file: ' + mu_fname)
        np.save(mu_fname, mu_list)
        np.save(sol_fname, rs_list)
        rs_list = []
        mu_list = []
    

#for k in range(100):
#    mu0 = Rom0._random_mu()
#    Rom0.run_rom(mu=mu0,evaluate=True)
#    Rom0.compare_rom_solution(mu0,interval=15,\
#                sol_plot_fname='rom_sol_' + str(k) + '.png',\
#                err_plot_fname='rom_err_' + str(k) + '.png')
