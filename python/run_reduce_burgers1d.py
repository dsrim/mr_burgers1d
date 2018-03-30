
from burgers1d import Rom
import numpy as np
import matplotlib.pyplot as pl
import os,sys

Rom0 = Rom()

Rom0.load_data()
Rom0._reduce_bases(tol=1e-6)
Rom0._rom_reduced=True


for k in range(5):
    mu0 = Rom0._random_mu()
    Rom0.run_rom(mu=mu0,evaluate=True)
    Rom0.compare_rom_solution(mu0)
    os.system('mv rom_error.png rom_error_' + str(k) + '.png')
    os.system('mv rom_solution.png rom_solution_' + str(k) + '.png')
#Rom0.mc_sample()
