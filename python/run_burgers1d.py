
from burgers1d import Rom
import numpy as np
import matplotlib.pyplot as pl
import os,sys



build_bases = True      # construct local bases using disp. interpolation
reduce_bases = True     # further reduce local bases

if build_bases:
    Rom0 = Rom()
    # run HFM and save plots
    Rom0.sample_hfm(N=250,dt=0.0125,xl=0.,xr=100.,M=4000)
    Rom0.plot_hfm_solution()
    
    # construct / save basis fctns  
    Rom0.build_bases(tol=1e-10,max_basis_size=200,Mfinal=3000,t_interval=20,\
                   nalpha=4,P=4)
    Rom0.save()

if reduce_bases:
    print('loading pickled rom...')
    with open('_output/rom_file.pkl',mode='r') as input_file:
        Rom0 = pkl.load(input_file)

    Rom0.load()
    Rom0.reduce_bases(tol=1e-6)
    Rom0.save()




Rom0.load()
for k in range(100):
    mu0 = Rom0.random_mu()
    Rom0.run_rom(mu=mu0,evaluate=True)
    Rom0.compare_rom_solution(mu0,interval=15,\
                sol_plot_fname='rom_sol_' + str(k) + '.png',\
                err_plot_fname='rom_err_' + str(k) + '.png')
