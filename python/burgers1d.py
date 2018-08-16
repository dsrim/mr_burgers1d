import sys
sys.path.insert(0, '../../mr/python')
import mr
import numpy as np
import matplotlib.pyplot as pl
from scipy.spatial import Delaunay
from scipy.interpolate import InterpolatedUnivariateSpline
from copy import copy


class Rom(object):
    r"""Basic object representing reduced-order models

    """


    def __init__(self):
        r"""
        ROM initialization routine.
        """

        self._N = None              # HFM dof 
        self._x = None              # HFM domain 
        self._h = None              # HFM mesh-width for HFM
        self._mu_current = None     # current set of parameters
        self._hfm_M = None          # HFM number of time-steps
        self._hfm_dt = None         # HFM time-step size 
        self._hfm_S = None          # HFM stiffness
        self._hfm_src = None        # HFM source-term 
        self._hfm_k2mu_list = None  # list of HFM parameters (mu)
        self._hfm_u_list = None     # list of HFM solutions

        self._rom_tri = None        # ROM Delaunay triangular of domain
        self._rom_P = None
        self._rom_r1 = None
        self._rom_ra = None
        self._rom_ra_list = []
        self._rom_i0 = None
        self._rom_r_list = []       # ROM solution list
        self._rom_basis = []        # ROM basis functions
        self._rom_basis_raw = []    # ROM basis functions
        self._rom_F = []            # ROM flux functions
        self._rom_bc = []           # ROM BC constraints
        self._rom_src = []          # ROM source functions
        self._rom_M = []            # ROM basis numbers
        self._rom_T = []            # ROM transition matrices
        self._time_index_list = [] # ROM grid-points in time

        self._rom_reduced = False   # ROM basis reduction flag
        self._rom_set_up = False   # ROM basis reduction flag

        self._dpi = 200             # dpi for savefigs

    @property
    def hfm_S(self):
        r"""stiffness matrix"""
        if self._hfm_S is None:
            self._compute_stiffness()
        return self._hfm_S

    @property
    def hfm_src(self):
        r"""stiffness matrix"""
        self._compute_hfm_src()
        return self._hfm_src

    def _compute_hfm_src(self):
        
        mu1 = self._current_mu[1]
        x = self._x

        self._hfm_src = 0.02*np.exp(mu1*x)

    def _compute_stiffness(self):
        r"""
        Assemble stiffness matrix

        """
        N = self._N
        h = self._h
        
        S = np.zeros((N,N))
        S += (np.diag([.5]*N,k=0) - np.diag([.5]*(N-1),k=-1)) / h
        S[ 0, 0] = -.5 /h
        S[ 0, 1] =  .5 /h
        S[-1,-2] = -.5 /h
        S[-1,-1] =  .5 /h

        self._hfm_S = S

        
    def _hfm_time_step(self,u):
        r"""
         Take one time-step of the high-fidelity model
        
        """
        
        S = self.hfm_S
        src = self.hfm_src
        dt = self._hfm_dt
        
        u = np.array(u)
        u = u - np.dot(dt*S,u*u) + dt*src
        
        return u


    def run_hfm(self,mu,N=250,xl=0.,xr=100.,dt=0.0125,M=4000):
        r"""
        Run High Fidelity Model (HFM)

        INPUTS
            mu: (2,)-array designating parameters

        """

        set_domain = False
        
        # set up domain
        if self._N == None:
            self._N = N
            set_domain = True
        #elif self._N != N:
        #    print('resetting N')
        #    self._N = N
        #    set_domain = True

        if set_domain:
            x = np.linspace(xl,xr,N)
            h = (xr - xl)/N
            
            self._x = x
            self._h = h
            self._hfm_dt = dt
            self._hfm_M  = M
        else:
            N = self._N
            x = self._x
            h = self._h
            dt = self._hfm_dt 
            M = self._hfm_M  
        
        self._current_mu = mu
        
    
        # solve one high-dimensional problem for given parameter mu
        u = np.zeros((N,M+1))    # initialize zero array for solution u

        
        for j in range(M):
            u[0:2] = 1.*mu[0]    # set incoming BC
            u[:,j+1] = self._hfm_time_step(u[:,j])
        
            if 0:
                # plot CDFs of u
                v = np.cumsum(u) 
                pl.plot(x,v,'b')

        return u

        
    def sample_hfm(self,N=250,dt=0.0125,M=4000,xl=0.,xr=100.):
        r"""
        run high fidelity model (HFM) in bulk
        
        """

        # Run HDM for give parameter values 

        # parameter values
        mu = np.zeros(2)
        
        mu0_list = [3.  , 6.  , 9.   ]    # incoming left BC
        mu1_list = [0.02, 0.05, 0.075]    # parameter for the source term

        self._hfm_m0_list = mu0_list
        self._hfm_m1_list = mu1_list

        # evaluation points (KC)
        # mu0_list = [4.5  ]
        # mu1_list = [0.038]

        k2mu   = []
        u_list = []
        k = 0
        mu = [0.,0.]
        
        for i in range(len(mu0_list)):
            for j in range(len(mu1_list)):
                
                mu[0] = mu0_list[i]
                mu[1] = mu1_list[j]
                
                k2mu.append((mu[0],mu[1]))
                print '= '\
                    + '  mu0 = ' + str(mu[0]) \
                    + ', mu1 = ' + str(mu[1])
                uk = self.run_hfm(mu,\
                                 N=N,xl=xl,xr=xr,dt=dt,M=M)
                u_list.append(uk)
                fname = '_output/hfm_sol_' + str(k) + '.npy'
                np.save(fname,uk)
                k += 1
        
        self._hfm_k2mu_list = k2mu
        self._hfm_u_list = u_list
        
        
    def _diffp(self,a):
        r"""
            differentiate with padding: (output vector length unchanged)
              
        """
        return np.diff(np.concatenate((a[:1],a)))


    def _supports(self,a):
        r"""
            return begin/ending indices of connected supports of a vector a

        """
        b = 1.*(a > 0.)
        dpb = np.diff(np.concatenate(([0.],b),axis=0))
        I = np.arange(len(a))
        ibegin = I[dpb > 0.]
        iend   = I[dpb < 0.]
        if len(ibegin) > len(iend):
            iend = np.concatenate((iend, [len(a)]))
        return (ibegin,iend)


    def _get_parts(self,v):
        r"""
            return positive / negative parts of a vector

        """
        vp = v*(v >= 0.)
        vm = v*(v <  0.)
        return vp,vm

    
    def _get_spos(self,v):
        r"""
            return mean position of the supports  

        """
        
        supp = np.abs(v) > 0.
        return np.mean(np.arange(len(v))[supp])

    def _get_pieces(self,v):
        r"""
            obtain pieces (supports) of a vector where the other pieces
            have been zero'ed out

        """
        ibegin,iend = self._supports(v)
        pieces_list = []
        w = None
        for k in range(len(ibegin)):
            ii = np.zeros(v.shape,dtype=bool)
            ii[ibegin[k]:iend[k]] = True
            w = v*ii
            pieces_list.append(w)
        return pieces_list


    def _add_gc(self,v,n=(2,2)):
        r"""
        pre- and post-pend ghost cells

        """
        return np.concatenate((np.array([0.]*n[0]),v,np.array([0.]*n[1])),axis=0)


    def _del_gc(self,v,n=(2,2)):
        r"""
            remove ghost cells

        """
        if (n[0] > 0) and (n[1] > 0):
            return v[n[0]:-n[1]]
        elif (n[0] == 0) and (n[1] > 0):
            return v[:-n[1]]
        elif (n[0] > 0) and (n[1] == 0):
            return v[n[0]:]


    def plot_all_parts(self,v0,v1):
        r"""
            plot all positive / negative parts

        """
        diffp = self._diffp
        get_parts = self._get_parts

        dv0 = diffp(v0)
        dv1 = diffp(v1)
    
        dv0p,dv0m = get_parts(dv0)
        dv1p,dv1m = get_parts(dv1)
    
        fig,axes = pl.subplots(nrows=3,ncols=2,figsize=(12,6))
        axes[0,0].plot(v0);   axes[0,0].set_title('u0');
        axes[0,1].plot(v1);   axes[0,1].set_title('u1');
        axes[1,0].plot(dv0p); axes[1,0].set_title('+ part of Du0');
        axes[2,0].plot(dv0m); axes[2,0].set_title('- part of Du0');
        axes[1,1].plot(dv1p); axes[1,1].set_title('+ part of Du1');
        axes[2,1].plot(dv1m); axes[2,1].set_title('- part of Du1');
        
        fig.tight_layout()

        return fig, axes


    def _plot0(self):

        # plot some positive and negative parts for illustration
        j0 = 3
        j1 = 200
        k  = 2
        
        print(u2mu[k])
        
        mu = np.zeros(2)
        mu[0] = u2mu[k][0]
        mu[1] = u2mu[k][1]
        
        u0   = u_list[k]
        u00  = np.array(u0)[:, j0]
        u01  = np.array(u0)[:, j1]
        
        du00 = diffp(u00)
        du01 = diffp(u01)
        
        du00p,du00m = get_parts(du00)
        du01p,du01m = get_parts(du01)
        
        plot_all_parts(u00,u01)


    def _dinterp_parts(self,v0,v1,alpha,tol=1e-12,verbose=False):
        # TODO: name dinterp_pieces makes more sense?
        
        # preamble
        restrict = False
        add_gc = self._add_gc
        del_gc = self._del_gc
        get_pieces = self._get_pieces
        get_spos = self._get_spos
        
        # extrapolate right BC

        # get given values
        y0 = v0[-4:]
        y1 = v1[-4:]
    
        # add ghost cells of size 2
        v0 = add_gc(v0,n=(0,2))
        v1 = add_gc(v1,n=(0,2))
        
        order = 3    # use cubic splines
        x0 = np.array([1,2,3,4])
        s0 = InterpolatedUnivariateSpline(x0, y0, k=order)
        s1 = InterpolatedUnivariateSpline(x0, y1, k=order)
        
        # set ghost cells
        v0[-2] = s0(5)
        v0[-1] = s0(6)
     
        v1[-2] = s1(5)
        v1[-1] = s1(6)
        
        # get pieces
        w0_list = get_pieces(v0)
        w1_list = get_pieces(v1)
        
        n0 = len(w0_list)    
        n1 = len(w1_list)

        count = 0
        restrict = 0      # a flag to remove ghost cells later
        gl = 0
        gr = 0
        
        while (n0 != n1):
            
            # if the number of pieces are different
            # match the odd one out with the support in the ghost cells
            

            
            if verbose:
                print('number of pieces: (n0,n1)= ' + str(n0) + ',' + str(n1))
            
            
            if n1 < n0:
                if verbose:
                    print('swapping v0 and v1..')
                
                n2 = n0
                n0 = n1
                n1 = n2
                
                w2_list = w0_list
                w0_list = w1_list
                w1_list = w2_list
                
                v2 = v0
                v0 = v1
                v1 = v2
                
                alpha = 1. - alpha
                if verbose:
                    print(r' -> new # pieces: (n0,n1)= ' + str(n0) + ',' + str(n1))
            
            # get medians of the supports
            spos0 = np.zeros(n0)
            spos1 = np.zeros(n1)
            
            for j,w in enumerate(w0_list):
                spos0[j] = get_spos(w)
            
            for j,w in enumerate(w1_list):
                spos1[j] = get_spos(w)
            
            # compute distance matrix
            dist_mat = np.zeros((n0,n1))
            for j0 in range(n0):
                for j1 in range(n1):
                    dist_mat[j0,j1] = np.abs(spos0[j0] - spos1[j1])
                    
            # match closer supports
            adj_mat  = np.zeros((n0,n1),dtype='bool')
            for j0 in range(n0):
                i = int(np.argmin(dist_mat[j0,:]))
                adj_mat[j0,i] = True
            
            ax0 = 0
            vn0 = v0
            vn1 = v1
            spos = spos1
            
            ii = ~adj_mat.sum(axis=ax0,dtype='bool')
            
            if verbose:
                print(dist_mat)
                print(adj_mat)
                print(ii)     
            
            S = np.mean(vn0)
            if S < 1e-5:
                S = 1e-5
            
            if n0 > 0:
                spos_ii = spos[ii]
                if len(spos_ii) > 1:
                    spos_ii = spos_ii[0]
            else:
                # edge case when one vector is null
                if verbose:
                    print('WARNING: one vector is null!')
                spos_ii = spos[0]
            
            if verbose:
                print(spos_ii)            

            if float(spos_ii) < (float(len(w)) / 2.):
                vn0 = add_gc(vn0,n=(2,0))
                vn0[0] = 1.
                vn1 = add_gc(vn1,n=(2,0))
                gl += 1
            else:
                vn0 = add_gc(vn0,n=(0,2))
                vn0[-1] = 1.
                vn1 = add_gc(vn1,n=(0,2))
                gr += 1
        
            restrict += 1
            
            # get parts
            v0 = vn0
            v1 = vn1

            w0_list = get_pieces(v0)
            w1_list = get_pieces(v1)

            n0 = len(w0_list)    
            n1 = len(w1_list)

            if verbose:
                print(r' -> added # pieces:  n0,n1 = ' + str(n0) +','+ str(n1))
                print(r'                     gl,gr = ' + str(gl) +','+ str(gr))

            count += 1
            if count > 5:
                print('WARNING: missing-piece unresolved')
                break
    
        g_list = []
        s_list = []
        
        for j in range(len(w0_list)):
            
            w0 = w0_list[j]
            w1 = w1_list[j]
                
            if np.linalg.norm(w0 - w1) < tol:
                # if the two vectors are same, do nothing.
                h = w0
                
            else:
                x = np.linspace(0.,1.,len(w0))
                
                s0 = np.sum(w0)
                s1 = np.sum(w1)
                    
                x1,x2,Fv,Fv2 = mr.computeFz(x,w0/s0,w1/s1);    # need not be computed every time
                xx,g = mr.dinterp(x,x1,x2,Fv,alpha);
                    
                s = (1. - alpha)*s0 + alpha*s1
                if np.sum(g) > 0.:
                    h = s*g/np.sum(g)
                else:
                    h = s*w0
                
            # remove ghost cells
            if restrict > 0:
                h = del_gc(h,n=(2*gl,2*gr))
    
            # remove right ghost cells
            h = del_gc(h,n=(0,2))
            if 0:
                f,axes = pl.subplots(nrows=3,figsize=(10,6))
                axes[0].plot(w0)
                axes[1].plot(w1)   
                axes[2].plot(h)
                axes[2].set_title(str(s))
                f.tight_layout()
    #                 print(str(s))
            
    
            g_list.append(h)
    
    
        return g_list

    def _plot1(self):
        
                
        # plot each piece
        fig,axes = pl.subplots(nrows=2,ncols=1,figsize=(8,4))
        supports_list = []
        
        p_list = get_pieces(du00p)
        axes[0].set_title('+ part')
        for w in p_list:
            axes[0].plot(x,w)
        
        for w in p_list:
            supports_list.append(w > 0.)
            x0 = x[w > 0.]
            zz = x0*0.
            axes[0].plot(x0,zz,'sr-',linewidth=2.,markersize=2.)
        
        p_list = get_pieces(-du00m)
        axes[1].set_title('- part')
        for w in p_list:
            supports_list.append(w > 0.)
            axes[1].plot(x,-w)
            x0 = x[w > 0.]
            zz = x0*0.
            axes[1].plot(x0,zz,'sr-',linewidth=2.,markersize=2.)
        
        fig.tight_layout()


    def _dinterp_all(self,v0p,v1p,v0m,v1m,alpha):

        dinterp_parts = self._dinterp_parts
        
        gp_list = dinterp_parts( v0p, v1p,alpha)
        gm_list = dinterp_parts(-v0m,-v1m,alpha)
    #     g = np.zeros(N)
    #     for gp in gp_list:
    #         g += gp
        for k,gm in enumerate(gm_list):
            gm_list[k] *= -1.
        return gp_list + gm_list


    def _get_cell(self,t,ts):
        N = len(ts)
        for j in range(N):
            if t < ts[j]:
                return j-1
        return 0

    def _get_W(self,v0,v1,\
               alphas,offsets=(0.,0.),return_parts=False,return_svd=True):
    
        dinterp_alphas = self._dinterp_alphas
        get_parts = self._get_parts
        get_pieces = self._get_pieces
        diffp = self._diffp
        
        us = dinterp_alphas(v0,v1,alphas,offsets=offsets)
        
        if return_parts:
            us_pieces = []
            #us_pieces += us    # add generated basis themselves
            for k,u in enumerate(us):
                dup,dum = get_parts(diffp(u))
                dup_list = get_pieces(dup)
                dum_list = get_pieces(-dum)
                for dup in dup_list:
                    supp = 1.*(np.abs(dup) > 0.)
                    us_pieces.append(np.cumsum(dup)*supp)
                    if 0:
                        pl.plot(np.cumsum(dup)*supp)
                        pl.savefig('dup_' + str(k))
                        pl.close()
                for dum in dum_list:
                    supp = 1.*(np.abs(dum) > 0.)
                    us_pieces.append(-np.cumsum(dum)*supp)
                    if 0:
                        pl.plot(np.cumsum(dum)*supp)
                        pl.savefig('dum_' + str(k))
                        pl.close()
            us_pieces.append(np.ones(us[-1].shape))
            us = np.array(us_pieces).T     
            
        else:
            dim = us[-1].shape
            us.append(np.ones(dims))
            for k in range(1,8):
                ubc0 = np.zeros(dims)
                ubc0[:k] = 1.
                us.append(ubc0.copy())
            us = np.array(us).T
            
        if return_svd:
            U,s,V = np.linalg.svd(us,full_matrices=0)
            return U,s,V,us
        else:
            return us


    def _dinterp_alphas(self,v0,v1,alphas,\
                       offsets=(0.,0.),return_parts=False,return_int=True):

        get_parts = self._get_parts
        dinterp_all = self._dinterp_all
        
        v0p,v0m = get_parts(v0)
        v1p,v1m = get_parts(v1)
        us = []
        
        for alpha in alphas:
            g_list = dinterp_all(v0p,v1p,v0m,v1m,alpha)    # can be made faster
            offset = (1-alpha)*offsets[0] + alpha*offsets[1]
            h = np.zeros(v0.shape)
            if not return_parts:
                for g0 in g_list:
                    if return_int:
                        h += np.cumsum(g0)
                    else:
                        h += g0
                us.append(h)
            else:
                for g0 in g_list:
                    if return_int:
                        us.append(np.cumsum(g0))
                    else:
                        us.append(g0)
        return us


#    def _precompute_flux_rom(self,U):
#        # pre-compute flux
#        N = U.shape[0]
#        m = U.shape[1]
#        F = np.zeros((m,m,m))
#        for i in range(m):
#            for j in range(m):
#                for l in range(m):
#                    F[i,j,l] -= .5\
#                             *np.sum(U[1:,i]*U[1:,j]*U[1:,l] - U[1:,i]*U[:(N-1),j]*U[:(N-1),l])
#                    F[i,j,l] -= .5\
#                             *np.sum(U[0,i]*U[1,j]*U[1,l] - U[0,i]*U[0,j]*U[0,l])
#        return F

    def _precompute_flux_rom(self,U):
        # pre-compute flux
        N = U.shape[0]
        m = U.shape[1]
        F = - .5*(np.einsum('ij,ik,il->jkl',U[1:,:],U[1:,:],U[1:,:])  \
                -  np.einsum('ij,ik,il->jkl',U[1:,:],U[:(N-1),:],U[:(N-1),:])) \
             - .5*(np.einsum('j,k,l->jkl',U[0,:],U[1,:],U[1,:]) \
                -  np.einsum('j,k,l->jkl',U[0,:],U[0,:],U[0,:])) 
        return F


    def _precompute_src_rom(self,U,p=20):
        
        s0 = 0.02    
        N = U.shape[0]      # HFM dof
        m = U.shape[1]      # ROM dof
        
        z = np.linspace(0.,100.,N) 
        V = np.zeros((m,p))
        w = np.ones(N)
    
        for j in range(1,p):
            w *= z / float(j)
            V[:,j] = s0*np.dot(U.T,w)
    
        V[:,0] = s0*np.dot(U.T,np.ones(N))
        
        return V


    def _rom_time_step(self,r,F,src,dt,mu1,bc):
        
        h = self._h
        p = src.shape[1]
        mu1_array = np.array([mu1 ** j for j in range(p)])
        r = np.array(r).flatten()
        dr = dt/h*np.dot(np.dot(F,r),r) + dt*np.dot(src,mu1_array)
        bc = bc / np.linalg.norm(bc)
        dr = dr - np.dot(dr,bc)*bc
        #r = r + dt/h*np.dot(np.dot(F,r),r) + dt*np.dot(src,mu1_array)
        r = r + dr
        
        return r


    def plot_triangulation(self):

        tri = self._rom_tri
        dpi = self._dpi
        
        nsimplex= tri.nsimplex        # no of triangles
        simplices = tri.simplices     # nodes of each triangle
        points = tri.points           # coordinates of vertices
        f, ax0 = pl.subplots(nrows=1,ncols=1)
        
        for j in range(nsimplex):
            
            inodes = simplices[j,[0,1,2,0]]
            
            xcoords = points[inodes,0]
            ycoords = points[inodes,1]
            ax0.plot(xcoords, ycoords,'-b.')
    
        ax0.set_xlabel('$\mu_1$')
        ax0.set_ylabel('$\mu_2$')

        f.savefig('triangulation.png',dpi=dpi)
        pl.close(f)

    def _get_barycentric(self,tri,x):
        
        ndim = tri.ndim
        n = int(tri.find_simplex(x))    # find simplex number
        T = tri.transform[n,:ndim,:ndim]
        r = tri.transform[n,ndim,:]
            
        c = np.dot(T, x-r)
        
        return c


    def _get_spatial_coords(self,tri,n,c):
    
        ndim = tri.ndim
        T = tri.transform[n,:ndim,:ndim]
        r = tri.transform[n,ndim,:]
        
        x = np.linalg.solve(T,c) + r

        return x

    def _get_uniform_samples(self,n):
        
        x = np.linspace(0.,1.,n)
        y = np.linspace(0.,1.,n)
        x0,y0 = np.meshgrid(x,y)
        x0 = x0.flatten()
        y0 = y0.flatten()
        
        return np.array([x0[(x0 + y0) <= 1.],y0[(x0 + y0) <= 1.]])    

    def _get_sides(self,tri,n):
        
        simplex = tri.simplices[n,:]
        side0 = [simplex[2], simplex[0]]
        side1 = [simplex[2], simplex[1]]
        
        return side0,side1

    def _get_time_index(self,j,times_list):
        l0 = next((i for i,i0 in enumerate(times_list) if j <i0),\
                   len(times_list)) - 1
        if 0:
            print(str(j) + ' , interval [' + str(times_list[l0]) + ', ' + str(times_list[l0+1]) + ']')
        return max([l0,0])
    
    
    def _dinterp2D(self,u10,u01,u00,c,offset=None,return_parts=False):
        r'''
        
        c is a 2-array with barycentric coordinates
        
        '''
        diffp = self._diffp        
        dinterp_alphas = self._dinterp_alphas

        v00 = diffp(u00)
        v10 = diffp(u10)
        v01 = diffp(u01)
        
        if offset == None:
            offset = (1. - c[0] - c[1])*u00[0] + c[0]*u10[0] + c[1]*u01[0]
        
        alphas = [c[0]]
        g_list = dinterp_alphas(v00,v10,alphas,return_int=False)
        va0 = g_list[0]
        
        g_list = dinterp_alphas(v01,v10,alphas,return_int=False)
        va1 = g_list[0]
        
    #     va0[-1] = va0[-2]  # kluging it
    #     va1[-1] = va1[-2]
        
    #     pl.figure(figsize=(14,2))
    #     pl.plot(va0)
    #     pl.plot(va0 - va1,'--')
    #     print(va0)
    #     w0p,w0m = get_parts(v10)
    #     w1p,w1m = get_parts(v01)
        
    #     pl.plot(w0p,'r')
    #     pl.plot(w1p,'b')
        
        if c[0] < 1.:
            alphas = [c[1] *(1. - c[0])]
        else:
            alphas = [0.]
        g_list = dinterp_alphas(va0,va1,alphas,return_int=True,return_parts=return_parts)
        if not return_parts:
            uaa = g_list[0]
            uaa += offset
    #     print(offset)
            return uaa
        else:
    #         uaa_list = []
    #         for g in g_list:
    #             uaa_list.append(np.cumsum(g))
            return g_list



    def _dinterp2D_tri(self,tri,u_list,snapshot_indices,mu):
        
        uaa_snapshot = []

        get_barycentric = self._get_barycentric
        get_sides = self._get_sides
        
        n = tri.find_simplex(mu)
        c = get_barycentric(tri,mu)
        c = np.abs(c)
        side0,side1 = get_sides(tri,n)
    
    
        for j in range(len(snapshot_indices)):
    
            tm = snapshot_indices[j]
            
            anm = np.array(u_list[side0[0]][:,tm]).flatten()
        
            u00 = np.array(u_list[side0[0]][:,tm]).flatten()
            u10 = np.array(u_list[side0[1]][:,tm]).flatten()
            u01 = np.array(u_list[side1[1]][:,tm]).flatten()
                
            if 0:
                fig,ax = pl.subplots(nrows=3,ncols=2,figsize=(15,2))
                
                u00p,u00m = get_parts(diffp(u00))
                u10p,u10m = get_parts(diffp(u10))
                u01p,u01m = get_parts(diffp(u01))
            
                ax[0,1].plot(u00,'b');
                ax[1,1].plot(u10,'g');
                ax[2,1].plot(u01,'r');  
        
                ax[0,0].plot(u00m,'b--')
                ax[1,0].plot(u10m,'g--')
                ax[2,0].plot(u01m,'r--')
            
            uaa = self._dinterp2D(u10,u01,u00,c,offset=0.,return_parts=False)
        
            if 0:
                pl.figure(figsize=(15,2))
                pl.plot(uaa)
                pl.title(str(j))
            
            uaa_snapshot.append(uaa)
        
        uaa_snapshot = np.array(uaa_snapshot).T
        return uaa_snapshot

    
    def build_bases(self,tol=1e-10,Mfinal=80,max_basis_size=200,\
                    t_interval=5,P=5,p = 40,nalpha=5,save_raw=False):
        r'''

        P = 5            no. of samples per triangular element

        '''


        # import internal functions
        diffp = self._diffp
        dinterp2D_tri = self._dinterp2D_tri
        get_W = self._get_W
        get_uniform_samples = self._get_uniform_samples
        get_barycentric = self._get_barycentric
        get_spatial_coords = self._get_spatial_coords
        precompute_flux_rom = self._precompute_flux_rom
        precompute_src_rom = self._precompute_src_rom

        # set up Delaunay triangulation
        mu0_list = self._hfm_m0_list
        mu1_list = self._hfm_m1_list
        
        Mu1,Mu0 = np.meshgrid(mu1_list,mu0_list)
        train_pts = np.concatenate(([Mu0.flatten()],[Mu1.flatten()]),axis=0).T
        tri = Delaunay(train_pts)
        self._rom_tri = tri
        
        self._rom_P = P
        z = get_uniform_samples(P)    # get uniformly-spaced samples

        # set up time-intervals for sampling
        time_index_list = [1,2] + range(3,Mfinal,t_interval)
        self._time_index_list = time_index_list
        
        if 0:
            # plot barycentric grid-points z
            fig,axes = pl.subplots(ncols=2,figsize=(10,4));
            axes[0].plot(z[0,:],z[1,:],'.');
            # plot parameter points in mu-plane
            for mu0 in mu_list:
                axes[1].plot(mu0[0],mu0[1],'r.');
        
        u_list = self._hfm_u_list      # get HFM solutions
        alphas = np.linspace(0.,1.,nalpha)  # number of points sampled in time

        for n in range(tri.nsimplex):
            
            print('building rom for element: ' + str(n))
            
            # get mu coordinates
            mu_list = []    # mu values for each sample points in z
            for j in range(z.shape[1]):
                c = np.array([z[0,j],z[1,j]])
                mu0 = get_spatial_coords(tri,n,c)
                mu_list.append(mu0)
            
            basis_list = []         # left singular vectors 
                                    # saved to external file unless specified
            basis_raw_list = []     # raw basis (not stored unless specified)
            s_list = []             # singular values at each slab
            M_list   = []           # number of basis kept at each slab
            basis_all = []          # all basis for the param-time-slab

            for j in range(len(time_index_list)-1):
                
                basis_j = []    # basis for time-slice
                print('- building basis at time-step n = ' \
                          + str(time_index_list[j]))
                
                for mu in mu_list:
                    
                    # TODO interp2D_tri also searches for the triangle for mu
                    uaa_snapshot = dinterp2D_tri(tri,u_list,time_index_list[j:(j+2)],mu)
                    uaa_j0 = uaa_snapshot[:,0]
                    uaa_j1 = uaa_snapshot[:,1]
                    
                    v0 = diffp(uaa_j0)
                    v1 = diffp(uaa_j1)
                    
                    us = get_W(v0,v1,alphas,\
                               return_parts=True, return_svd=False)
                    basis_j.append(us)
                
                basis_all.append(basis_j)


            for j in range(len(time_index_list)-1):
            
                if j == 0:
                    basis_j3 = basis_all[j] + basis_all[j+1]
                else:
                    basis_j3 = basis_all[j] 
                
                basis_array_j = np.concatenate(basis_j3, axis=1)
                U,s,V = np.linalg.svd(basis_array_j,full_matrices=0)
                if 0:
                    pl.semilogy(s,'-rx');
                s_list.append(s)
                M_k0 = \
                    next((i for i,si in enumerate(s/s[0]) if si < tol),100)
                M_k = min([M_k0, max_basis_size])
                M_list.append(M_k)
                Usmall = U[:,:M_k] 
                basis_list.append(Usmall)
                
                if save_raw:
                    basis_raw_list.append(basis_array_j)

            self._rom_basis.append(basis_list)
            if save_raw:
                self._rom_basis_raw.append(basis_raw_list)

            # off-line computations for flux and source
            F_list   = []
            src_list = []
            bc_list  = []
            
            K = len(basis_list)-1
            for k,U in enumerate(basis_list):
                
                M_k = M_list[k]
                print('precomputing flux and src for ' + str(k) + ' / ' \
                         + str(K) + ' | number of basis: ' + str(M_k))
                
                # compute reduced basis 
                F = precompute_flux_rom(U[:,:M_k])
                F_list.append(F)
            
                # compute source term
                src = precompute_src_rom(U[:,:M_k],p=p)
                src_list.append(src)

                # compute boundary conditions
                bc_list.append( U[0,:M_k] )
                print('= done')

            #TODO: following 4 lines necessary?
            self._rom_F.append(F_list)
            self._rom_M.append(M_list)
            self._rom_bc.append(bc_list)
            self._rom_src.append(src_list)

            # compute transition matrix
            transition_list = []
            
            for k in range(len(basis_list)-1):
            
                s0 = s_list[k]
            
                M_k   = M_list[k]
                M_kp1 = M_list[k+1]
            
                U_k   = basis_list[k]
                U_kp1 = basis_list[k+1]
            
                transition_list.append(\
                             np.dot(U_kp1[:,:M_kp1].T,U_k[:,:M_k]) )

            self._rom_T.append(transition_list)

            ##                                              ##
            ##      save generated lists to output files    ##
            ##                                              ##
            L = len(basis_list)
            for k in range(L):
                m = L - 1 - k
                
                Usmall = basis_list.pop()
                F = F_list.pop()
                src = src_list.pop()
                bc = bc_list.pop()
                
                basis_fname = \
                        '_output/basis_' +str(n)+ '_' +str(m)+ '.npy'
                F_fname = \
                        '_output/F_' +str(n)+ '_' +str(m)+ '.npy'
                src_fname = \
                        '_output/src_' +str(n)+ '_' +str(m)+ '.npy'
                bc_fname = \
                        '_output/bc_' +str(n)+ '_' +str(m)+ '.npy'
                
                np.save(basis_fname, Usmall)
                np.save(F_fname, F)
                np.save(src_fname, src)
                np.save(bc_fname, bc)
            
            for k in range(L-1):
                m = L-1 - k
                T = transition_list.pop()
                T_fname = \
                        '_output/transition_'+str(n)+'_'+str(m) + '.npy'
                np.save(T_fname, T)

            M_fname = '_output/M_' + str(n) + '.npy'
            np.save(M_fname,M_list)


    def mc_sample(self,nsols=1000,M0=4000):
        r"""
            further reduction of the model based on sampling
        """

        np.random.seed(12345)

        # generate samples
        for n in range(self._rom_tri.nsimplex):
            j = 0
            random_mu_list = []
            random_mu = self._random_mu
            rs_list = []

            while(len(random_mu_list) < nsols):
                # sample random variable
                mu0 = random_mu()
                
                n0 = int(self._rom_tri.find_simplex(mu0))
                if n0 == n:
                    random_mu_list.append(mu0)
            
                    print('running rom..')
                    self.run_rom(mu=mu0,frugal=True,M0=M0)
                    print('= done')
                    rs_list.append(copy(self._rom_r_list))
                    print('sample number: ' + str(j))
                    j += 1
            
            mu_fname = '_output/sampled_mu_' + str(n) + '.npy'
            sols_fname = '_output/sampled_rom_sols_' + str(n) + '.npy'
            np.save(mu_fname,random_mu_list)
            np.save(sols_fname,rs_list)
            self._rom_set_up = False

    def _dot3d(self,A,B,C,D):
        r"""
            compute A_{i,j,k} B_{i,I} C_{j,J} D_{k,K}
            resulting in: an array E of size {I,J,K}
        """

        E = np.dot(A,D)
        E = np.transpose(np.dot(np.transpose(E,(0,2,1)),C),(0,2,1))
        E = np.transpose(np.dot(np.transpose(E,(2,1,0)),C),(2,1,0))

        return E

    def _reduce_bases(self,tol=1e-6,max_nbasis=150,max_time_step=2000):
        r"""
            compute further reduction using low-dimensional space 
        
        """
        N = self._rom_tri.nsimplex
        til = self._time_index_list.tolist()
        M = len(til)
        
        for n in range(N):
            # compute reduction
            mu_fname = '_output/sampled_mu_' + str(n) + '.npy'
            sols_fname = '_output/sampled_rom_sols_' + str(n) + '.npy'

            M_fname = '_output/M_' + str(n) + '.npy'
            M_list = np.load(M_fname)
            snapshot_list = np.load(sols_fname)
            M = snapshot_list.shape[1]      # number of time intervals
            rM_list = []

            for m in range(len(M_list)-1):
                
                m1 = M_list[m]
                sampled_sols_list = []
                for i in range(til[m]-1,til[m+1]-1):
                    if i < max_time_step-1:
                        new_array = np.array([snapshot_list[:,i][k].T \
                            for k in range(snapshot_list.shape[0])]).T
                        sampled_sols_list.append(new_array)   
                
                if len(sampled_sols_list) == 0:
                    break
                sampled_sols = np.concatenate(sampled_sols_list,axis=1)
                print(sampled_sols.shape)
                
                M0 = sampled_sols.shape[1]      # number of "generated" bases
                W,s,V = np.linalg.svd(sampled_sols,full_matrices=0)

                # compute reduced basis by truncating singular values,
                # store no. of reduced basis
                m0 = \
                    next((i for i,si in enumerate(s/s[0]) if si < tol),\
                         max_nbasis)
                print('m = ' +str(m)+ ' | no. of reduced basis m0 = ' \
                    + str(m0) + ' / m1 = ' + str(m1))
                rM_list.append(m0)  
                W = W[:,:m0]
                
                # compute basis
                print('- computing basis...')
                basis_fname = \
                        '_output/basis_' +str(n)+ '_' +str(m)+ '.npy'
                basis = np.load(basis_fname)
                #m2 = min([basis.shape[1], W.shape[0]])
                rbasis = np.dot(basis[:,:m1],W)
                rbasis_fname = \
                        '_output/rbasis_' +str(n)+ '_' +str(m)+ '.npy'
                np.save(rbasis_fname,rbasis)

                # reduce flux
                print('- computing flux...')
                F_fname = \
                        '_output/F_' +str(n)+ '_' +str(m)+ '.npy'
                F = np.load(F_fname)
                rF = self._dot3d(F,W,W,W)   # faster than einsum
                
                rF_fname = \
                        '_output/rF_' +str(n)+ '_' +str(m)+ '.npy'
                np.save(rF_fname,rF)
                
                # reduce source
                print('- computing source...')
                src_fname = \
                        '_output/src_' +str(n)+ '_' +str(m)+ '.npy'
                src = np.load(src_fname)
                p = src.shape[1]
                #rsrc = np.zeros((m0,p))
                #for i in range(m0):
                #    rsrc[i,:] = np.dot(W[:,i],src)
                rsrc = np.dot(W.T,src)
                rsrc_fname = \
                        '_output/rsrc_' +str(n)+ '_' +str(m)+ '.npy'
                np.save(rsrc_fname,rsrc)
                
                # reduce BCs
                print('- computing reduced BC...')
                #bc_fname = \
                        #'_output/bc_' +str(n)+ '_' +str(m)+ '.npy'
                #bc = np.load(bc_fname)
                #rbc = np.dot(bc,W)
                rbc = rbasis[0,:m0]
                rbc /= np.linalg.norm(rbc)
                rbc_fname = \
                        '_output/rbc_' +str(n)+ '_' +str(m)+ '.npy'
                np.save(rbc_fname,rbc)

                # reduce transition list
                if (m > 0):
                    print('- computing transition matrix...')
                    T_fname =  \
                        '_output/transition_'+str(n)+'_'+str(m) + '.npy'
                    T = np.load(T_fname)
                    Tw = np.dot(T,W_old)
                    rT = np.dot(W.T,Tw)
                    rT_fname =  \
                        '_output/rtransition_'+str(n)+'_'+str(m) + '.npy'
                    np.save(rT_fname,rT)
                W_old = W                    
                print('= done.')
        
            rM_fname = \
                       '_output/rM_' + str(n) + '.npy'
            np.save(rM_fname, rM_list)
        self._rom_reduced = True
    

    def _random_mu(self):
        mu = np.random.rand(2)
        mu[0] = mu[0]*(9. - 3.) + 3.
        mu[1] = mu[1]*(0.075 - 0.02) + 0.02
        return mu

    def _set_up_rom(self,reset=False,load_basis=False,\
                         simplex_list=None, time_interval_list=None, M0=4000):
        r"""
        
        Load bases / F / src / transition_list into memory

        """

        if reset:
            self._rom_set_up = False

        if not self._rom_set_up:
            
            if self._rom_reduced:
                prefix = 'r'
            else:
                prefix = ''
            
            K = self._rom_tri.nsimplex
            M = len(self._time_index_list) 

            if simplex_list == None:
                k_list = range(K)
            else:
                k_list = simplex_list

            if time_interval_list == None:
                #TODO: not to be confused with usual time_index_list
                m_list = range(M-1)
            else:
                m_list = time_interval_list

            self._rom_basis = []
            self._rom_F = []
            self._rom_src = []
            self._rom_bc = []
            self._rom_T = []
            self._rom_M = []
            
            for k in k_list:
                M_fname = '_output/' + prefix + \
                        'M_' + str(k) + '.npy'
                M_list = np.load(M_fname)
                transition_list = []
                F_list = []
                src_list = []
                basis_list = []
                bc_list = []
                for m in m_list:
                    basis_fname = '_output/' + prefix + \
                            'basis_' +str(k)+ '_' +str(m)+ '.npy'
                    F_fname = '_output/' + prefix + \
                            'F_' +str(k)+ '_' +str(m)+ '.npy'
                    src_fname = '_output/' + prefix + \
                            'src_' +str(k)+ '_' +str(m)+ '.npy'
                    T_fname = '_output/' + prefix + \
                            'transition_'+str(k)+'_'+str(m+1) + '.npy'
                    bc_fname = '_output/' + prefix + \
                            'bc_' +str(k)+ '_' +str(m)+ '.npy'
    
                    F = np.load(F_fname)
                    src = np.load(src_fname)
                    bc = np.load(bc_fname)
                    
                    # load basis only if told to evaluate
                    # TODO: .copy() necessary?
                    if (m == 0) or ((m > 0) and load_basis):
                        basis = np.load(basis_fname)
                        basis_list.append(basis.copy())
                    F_list.append(F.copy())
                    src_list.append(src.copy())
                    bc_list.append(bc.copy())
                    
                    # transition list of length L-2 
                    if (m < M-2):
                        T = np.load(T_fname)
                        transition_list.append(T.copy())
                
                self._rom_M.append(M_list)
                self._rom_F.append(F_list)
                self._rom_src.append(src_list)
                self._rom_basis.append(basis_list)
                self._rom_bc.append(bc_list)
                self._rom_T.append(transition_list)
                
                self._rom_set_up = True
        else:
            print('ROM already set up')
            pass

#    def reduce_rom(self):
#
#        self.run_rom(mu=mu0)
    
    def run_rom(self,mu=[7.5, 0.035],M0=4000,\
                     evaluate=False,verbose=False,reread=False,\
                     frugal=False):
        r"""
        run reduced order model
        kwargs
          - mu: 2D-array [mu1,mu2]
          - M0: maximal number of time-steps
          - evaluate: represent in HFM basis and store in self._rom_ra_list

        """
        
        show_plots = False
        get_time_index = self._get_time_index        

        tri = self._rom_tri
        
        k = tri.find_simplex(mu)        # triangle number

        time_index_list = self._time_index_list
        L = len(time_index_list)
        
        if frugal:
            self._set_up_rom(simplex_list=[k])
            
            M_list = self._rom_M[0]
            F_list = self._rom_F[0]
            src_list = self._rom_src[0]
            basis_list = self._rom_basis[0]
            bc_list = self._rom_bc[0]
            transition_list = self._rom_T[0]
        else:
            self._set_up_rom(reset=reread,load_basis=evaluate,M0=M0)
    
            M_list = self._rom_M[k]
            F_list = self._rom_F[k]
            src_list = self._rom_src[k]
            basis_list = self._rom_basis[k]
            bc_list = self._rom_bc[k]
            transition_list = self._rom_T[k]
        
        #TODO: create setup_rom() to set up the reduced order model
        #      load transition_list, F_list, src_list, basis_list into memory
        # load basis/flux/src over time
        
        rom_time_step = self._rom_time_step

        M_0 = M_list[0]
        U = basis_list[0][:,:M_0]

        # input parameters
        mu0 = mu[0]
        mu1 = mu[1]

        # set up initial condition
        x = self._x
        u0a = np.zeros(x.shape)
        u0a[0:2] = mu0   # mu0
        r0 = np.dot(U.T,u0a)
        r1 = r0
        r1 = np.array(r1)       # redundant
        r_list = []

        if 0:
            pl.figure(figsize=(14,2))
            pl.plot(np.dot(basis_list[k][:,:M_0],r0))
        
        sc = 1
        dt = self._hfm_dt
        dt1 = dt/sc
        M0 = min([M0,time_index_list[-1]])

        if evaluate:
            self._rom_ra_list = []
            self._rom_ra_list.append(u0a)
        
        i0_old = get_time_index(0, time_index_list)

        for j in range(1,int(M0*sc)):
            i0 = get_time_index(j, time_index_list)
            if i0_old < i0:
                if verbose:
                    print((j,i0_old,i0))
                i0_old = i0
                r1 = np.dot(transition_list[i0-1],r1)
            
            F   =   F_list[i0_old]
            src = src_list[i0_old]
            bc = bc_list[i0_old]
            r1 = rom_time_step(r1,F,src,dt,mu1,bc)
            r1 = np.array(r1).flatten() 
            
            r_list.append(r1.copy())
            
            if evaluate:
                M_k = M_list[i0_old]
                if self._rom_reduced:
                    fname = '_output/rbasis_' + str(k) + '_' + str(i0_old) + '.npy'
                else:
                    fname = '_output/basis_' + str(k) + '_' + str(i0_old) + '.npy'
                Usmall = basis_list[i0_old]
                ua = np.dot(Usmall,r1)
                self._rom_ra_list.append(ua)
                self._rom_ra = ua
        
        self._rom_r1 = r1
        self._rom_r_list = r_list
        self._rom_i0 = i0_old

    def plot_rom_solution(self):

        ra_list = self._rom_ra_list
        L = len(ra_list)
        x = self._x
        dpi = self._dpi

        f,ax0 = pl.subplots(nrows=1,ncols=1,figsize=(10,3))
        for j in range(0,L,10):
            ra = ra_list[j]
            ax0.plot(x,ra,'royalblue')

        f.savefig('rom_solution.png',dpi=dpi)
        pl.close(f)


    def compare_rom_solution(self,mu,interval=10,\
                             sol_plot_fname='rom_solution.png',\
                             err_plot_fname='rom_error.png'):

        ra_list = self._rom_ra_list
        L = len(ra_list)
        x = self._x
        dpi = self._dpi

        f0,ax0 = pl.subplots(nrows=1,ncols=1,figsize=(10,3))
        f1,ax1 = pl.subplots(nrows=1,ncols=1,figsize=(10,3))
        u = self.run_hfm(mu)
        for j in range(0,L,interval):
            ra = ra_list[j]
            ax0.plot(x,ra,'royalblue')
            ax0.plot(x,u[:,j],'--r')
            
            ax1.plot(x,ra - u[:,j],'royalblue')

        # TODO save to _plots subdir?
        k = self._rom_tri.find_simplex(mu)
        ax0.set_title('element = ' + str(k))
        ax1.set_title('element = ' + str(k))
        f0.savefig(sol_plot_fname,dpi=dpi)
        f1.savefig(err_plot_fname,dpi=dpi)
        
        pl.close(f0)
        pl.close(f1)


    def plot_hfm_solution(self):

        u_list = self._hfm_u_list
        N = len(u_list)
        M = self._hfm_M
        x = self._x
        k2mu_list = self._hfm_k2mu_list
        K = 150
        dpi = self._dpi

        f,ax0 = pl.subplots(nrows=1,ncols=1,figsize=(10,3))
        for k in range(N):
            uk = u_list[k]
            mu = k2mu_list[k]
            for j in range(0,M,K):
                ax0.plot(x,uk[:,j],'royalblue')
            #ax0.set_title('HFM solution \n $(\mu_1,\mu_2)$ = (' \
                            #+ str(mu[0]) + ',' + str(mu[1]) + ')')
            ax0.set_title('$(\mu_1,\mu_2)$ = (' \
                            + str(mu[0]) + ',' + str(mu[1]) + ')')
            fname = '_output/sol_u_' + str(k) + '.png'
            #f.tight_layout()
            f.savefig(fname,dpi=dpi)
            ax0.clear()
        pl.close(f)


    def save_as_pickle(self,fname='_output/rom_file.pkl'):

        import pickle

        with open(fname, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    def save_data(self):
        r"""
        save ROM data
        """
        
        import pickle

        # to save in text-file
        with open('_output/info.txt',mode='w') as text_file:
            text_file.write('N  = {:10d}\n'.format(self._N))
            text_file.write('h  = {:26.15f}\n'.format(self._h))
            text_file.write('M  = {:10d}\n'.format(self._hfm_M))
            text_file.write('dt = {:26.15f}\n'.format(self._hfm_dt))
            text_file.write('P  = {:10d}\n'.format(self._rom_P))

        # to save as separate files

        # spatial grid
        fname = '_output/x.npy'
        np.save(fname,self._x)

        # time_index_list
        fname = '_output/time_index.npy'
        np.save(fname,self._time_index_list)
        
        # pickle triangulation
        fname = '_output/tri.pkl'
        with open(fname, 'wb') as output:
            pickle.dump(self._rom_tri, output, pickle.HIGHEST_PROTOCOL)


    def load_data(self):
        r"""
            
            load data 
            
        """
        
        import pickle

        with open('_output/info.txt',mode='r') as input_file:
            for row in input_file:
                val = row.split('=')[1]
                if 'N' == row.split('=')[0][0]:
                    N = int(val)
                if 'M' == row.split('=')[0][0]:
                    M = int(val)
                elif 'h' == row.split('=')[0][0]:
                    h = float(val)
                elif 'dt' == row.split('=')[0][0:2]:
                    dt = float(val)
                elif 'P' == row.split('=')[0][0]:
                    P = int(val)

        self._N = N
        self._h = h
        self._hfm_M = M
        self._hfm_dt = dt
        self._rom_P = P

        # spatial grid
        fname = '_output/x.npy'
        self._x = np.load(fname)

        # time_index_list
        fname = '_output/time_index.npy'
        self._time_index_list = np.load(fname)
        
        # pickle triangulation
        fname = '_output/tri.pkl'
        with open(fname, 'r') as input_pkl:
            self._rom_tri = pickle.load(input_pkl)

        
