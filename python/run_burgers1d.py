
from burgers1d import Rom
import numpy as np
import matplotlib.pyplot as pl
import os,sys
import pickle as pkl



build_basis = True       # construct local bases using disp. interpolation
reduce_basis = True      # further reduce local bases

if build_basis:
    Rom0 = Rom()
    # run HFM and save plots
    Rom0.sample_hfm(N=250,dt=0.0125,xl=0.,xr=100.,M=4000)
    Rom0.plot_hfm_solution()
    
    # construct / save basis fctns  
    Rom0.build_basis(tol=1e-10,max_basis_size=300,Tfinal=1000,t_interval=20,\
                   Pt=4,P=4)
    Rom0.save()

if reduce_basis:
    with open('_output/ROM_object.pkl',mode='r') as input_file:
        Rom0 = pkl.load(input_file)

    Rom0.load()
    Rom0._rom_reduced=False
    Rom0.reduce_basis(tol=1e-8,nsols=100,M0=60)
    Rom0.save()

with open('_output/ROM_object.pkl',mode='r') as input_file:
    Rom0 = pkl.load(input_file)

Rom0.load()
Rom0._rom_set_up = False
for k in range(10):
    mu0 = Rom0.random_mu()
    Rom0.run_rom(mu=mu0,evaluate=True,verbose=True)
    Rom0.compare_rom_solution(mu0,interval=20,\
                sol_plot_fname='rom_sol_{:05d}.png'.format(k),\
                err_plot_fname='rom_err_{:05d}.png'.format(k))
