
from burgers1d import Rom
import numpy as np
import matplotlib.pyplot as pl
import os,sys
import pickle as pkl



build_bases = False       # construct local bases using disp. interpolation
reduce_bases = False      # further reduce local bases

if build_bases:
    Rom0 = Rom()
    # run HFM and save plots
    Rom0.sample_hfm(N=250,dt=0.0125,xl=0.,xr=100.,M=4000)
    Rom0.plot_hfm_solution()
    
    # construct / save basis fctns  
    Rom0.build_bases(tol=1e-10,max_basis_size=200,Mfinal=1000,t_interval=20,\
                   Pt=4,P=4)
    Rom0.save()

if reduce_bases:
    print('loading pickled ROM...')
    with open('_output/ROM_object.pkl',mode='r') as input_file:
        Rom0 = pkl.load(input_file)

    Rom0.load()
    Rom0._rom_reduced=False
    Rom0.reduce_bases(tol=1e-8,nsols=1000,M0=60)
    Rom0.save()


print('loading pickled ROM...')
with open('_output/ROM_object.pkl',mode='r') as input_file:
    Rom0 = pkl.load(input_file)

Rom0.load()
for k in range(100):
    mu0 = Rom0.random_mu()
    #Rom0.run_rom(mu=mu0,evaluate=False,M0=60)
    Rom0.run_rom(mu=mu0,evaluate=True,M0=45,verbose=True)
    Rom0.compare_rom_solution(mu0,interval=15,\
                sol_plot_fname='rom_sol_{:05d}.png'.format(k),\
                err_plot_fname='rom_err_{:05d}.png'.format(k))
