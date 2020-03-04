#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "datadef.h"
#include "init.h"
#include "alloc.h"
#include "mpi.h"

#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))

extern int *ileft, *iright;
extern int nprocs, proc;

/* Computation of tentative velocity field (f, g) */
void compute_tentative_velocity(float **u, float **v, float **f, float **g,
    char **flag, int imax, int jmax, float del_t, float delx, float dely,
    float gamma, float Re)
{
    int  i, j;
    float du2dx, duvdy, duvdx, dv2dy, laplu, laplv;

    for (i=1; i<=imax-1; i++) {
        for (j=1; j<=jmax; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                du2dx = ((u[i][j]+u[i+1][j])*(u[i][j]+u[i+1][j])+
                    gamma*fabs(u[i][j]+u[i+1][j])*(u[i][j]-u[i+1][j])-
                    (u[i-1][j]+u[i][j])*(u[i-1][j]+u[i][j])-
                    gamma*fabs(u[i-1][j]+u[i][j])*(u[i-1][j]-u[i][j]))
                    /(4.0*delx);
                duvdy = ((v[i][j]+v[i+1][j])*(u[i][j]+u[i][j+1])+
                    gamma*fabs(v[i][j]+v[i+1][j])*(u[i][j]-u[i][j+1])-
                    (v[i][j-1]+v[i+1][j-1])*(u[i][j-1]+u[i][j])-
                    gamma*fabs(v[i][j-1]+v[i+1][j-1])*(u[i][j-1]-u[i][j]))
                    /(4.0*dely);
                laplu = (u[i+1][j]-2.0*u[i][j]+u[i-1][j])/delx/delx+
                    (u[i][j+1]-2.0*u[i][j]+u[i][j-1])/dely/dely;
   
                f[i][j] = u[i][j]+del_t*(laplu/Re-du2dx-duvdy);
            } else {
                f[i][j] = u[i][j];
            }
        }
    }

    for (i=1; i<=imax; i++) {
        for (j=1; j<=jmax-1; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
                duvdx = ((u[i][j]+u[i][j+1])*(v[i][j]+v[i+1][j])+
                    gamma*fabs(u[i][j]+u[i][j+1])*(v[i][j]-v[i+1][j])-
                    (u[i-1][j]+u[i-1][j+1])*(v[i-1][j]+v[i][j])-
                    gamma*fabs(u[i-1][j]+u[i-1][j+1])*(v[i-1][j]-v[i][j]))
                    /(4.0*delx);
                dv2dy = ((v[i][j]+v[i][j+1])*(v[i][j]+v[i][j+1])+
                    gamma*fabs(v[i][j]+v[i][j+1])*(v[i][j]-v[i][j+1])-
                    (v[i][j-1]+v[i][j])*(v[i][j-1]+v[i][j])-
                    gamma*fabs(v[i][j-1]+v[i][j])*(v[i][j-1]-v[i][j]))
                    /(4.0*dely);

                laplv = (v[i+1][j]-2.0*v[i][j]+v[i-1][j])/delx/delx+
                    (v[i][j+1]-2.0*v[i][j]+v[i][j-1])/dely/dely;

                g[i][j] = v[i][j]+del_t*(laplv/Re-duvdx-dv2dy);
            } else {
                g[i][j] = v[i][j];
            }
        }
    }

    /* f & g at external boundaries */
    for (j=1; j<=jmax; j++) {
        f[0][j]    = u[0][j];
        f[imax][j] = u[imax][j];
    }
    for (i=1; i<=imax; i++) {
        g[i][0]    = v[i][0];
        g[i][jmax] = v[i][jmax];
    }
}


/* Calculate the right hand side of the pressure equation */
void compute_rhs(float **f, float **g, float **rhs, char **flag, int imax,
    int jmax, float del_t, float delx, float dely)
{
    int i, j;

    for (i=1;i<=imax;i++) {
        for (j=1;j<=jmax;j++) {
            if (flag[i][j] & C_F) {
                /* only for fluid and non-surface cells */
                rhs[i][j] = (
                             (f[i][j]-f[i-1][j])/delx +
                             (g[i][j]-g[i][j-1])/dely
                            ) / del_t;
            }
        }
    }
}


void communicate_1d(float **p, int imax, int jmax, int p_istart, int p_iend, int rank, int size)
{
    // MPI communication
    if (size == 1) {return;}
    if (rank == 0) {
        // printf("Sending Columns %d and receiving %d from process %d of %d\n", p_iend - 1, p_iend, rank, size);
        MPI_Send(p[p_iend - 1], jmax + 2, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
        MPI_Recv(p[p_iend], jmax + 2, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else if (rank == size - 1) {
        // printf("Sending Columns %d and receiving %d from process %d of %d\n", p_istart, p_istart - 1, rank, size);
        MPI_Send(p[p_istart], jmax + 2, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD);
        MPI_Recv(p[p_istart - 1], jmax + 2, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        // printf("Sending Columns %d and %d and receiving %d and %d from process %d of %d\n", p_iend - 1, p_istart, p_iend, p_istart - 1, rank, size);
        MPI_Send(p[p_iend - 1], jmax + 2, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
        MPI_Send(p[p_istart], jmax + 2, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD);

        MPI_Recv(p[p_iend], jmax + 2, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(p[p_istart - 1], jmax + 2, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}


/* Red/Black SOR to solve the poisson equation */
int poisson(float **p, float **rhs, char **flag, int imax, int jmax,
    float delx, float dely, float eps, int itermax, float omega,
    float *res, int ifull, int rank, int size)
{
    int i, j, iter;
    float add, beta_2, beta_mod;
    float p0 = 0.0;
    float local_residual = 0.0;
    
    int rb; /* Red-black value. */
    
    // Divide p between ranks
    // istart is inclusive and iend is exclusive
    int p_istart, p_iend;
    int p_iwidth = (imax + 2) / size;

    if (rank != size - 1) {
        p_istart = rank * p_iwidth;
        p_iend = (rank + 1) * p_iwidth;
    } else {
        p_istart = (size - 1) * p_iwidth;
        p_iend = imax + 2;
    }
    float **p_copy = alloc_floatmatrix(imax + 2, jmax + 2);
    memcpy(p_copy[0], p[0], (imax + 2) * (jmax + 2));

    // check copy
    // printf("copy: %f || original: %f\n", p[5][5], p_copy[5][5]);
    // printf("copy: %f || original: %f\n", p[2][9], p_copy[2][9]);
    // printf("copy: %f || original: %f\n", p[7][3], p_copy[7][3]);

    float rdx2 = 1.0/(delx*delx);
    float rdy2 = 1.0/(dely*dely);
    beta_2 = -omega/(2.0*(rdx2+rdy2));

    float partial_sum = 0.0;
    /* Calculate sum of squares */
    for (i = max(p_istart, 1); i <= min(p_iend - 1, imax); i++) {
    //for (i = 1; i <= imax; i++) {
        for (j=1; j<=jmax; j++) {
            if (flag[i][j] & C_F) { partial_sum += p[i][j]*p[i][j]; }
        }
    }
    MPI_Allreduce(&partial_sum, &p0, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
   
    p0 = sqrt(p0/ifull);
    if (p0 < 0.0001) { p0 = 1.0; }
    printf("p0: %f in process %d || ", p0, rank);

    /* Red/Black SOR-iteration */
    for (iter = 0; iter < itermax; iter++) {
        for (rb = 0; rb <= 1; rb++) {
            // for (i = max(1, p_istart); i <= min(imax, p_iend - 1); i++) {
            for (i = 1; i <= imax; i++) {
                for (j = 1; j <= jmax; j++) {

                    if ((i+j) % 2 != rb) { continue; }
                    if (flag[i][j] == (C_F | B_NSEW)) {
                        /* five point star for interior fluid cells */
                        p[i][j] = (1.-omega)*p[i][j] - 
                              beta_2*(
                                    (p[i+1][j]+p[i-1][j])*rdx2
                                  + (p[i][j+1]+p[i][j-1])*rdy2
                                  -  rhs[i][j]
                              );
                    } else if (flag[i][j] & C_F) { 
                        /* modified star near boundary */
                        beta_mod = -omega/((eps_E+eps_W)*rdx2+(eps_N+eps_S)*rdy2);
                        p[i][j] = (1.-omega)*p[i][j] -
                            beta_mod*(
                                  (eps_E*p[i+1][j]+eps_W*p[i-1][j])*rdx2
                                + (eps_N*p[i][j+1]+eps_S*p[i][j-1])*rdy2
                                - rhs[i][j]
                            );
                    }
                } /* end of j */
            } /* end of i */
            // communicate_1d(p, imax, jmax, p_istart, p_iend, rank, size);
        } /* end of rb */
        
        // Parallel execution with the copy
        for (rb = 0; rb <= 1; rb++) {
            for (i = max(1, p_istart); i <= min(imax, p_iend - 1); i++) {
                for (j = 1; j <= jmax; j++) {

                    if ((i+j) % 2 != rb) { continue; }
                    if (flag[i][j] == (C_F | B_NSEW)) {
                        /* five point star for interior fluid cells */
                        p_copy[i][j] = (1.-omega)*p_copy[i][j] - 
                              beta_2*(
                                    (p_copy[i+1][j]+p_copy[i-1][j])*rdx2
                                  + (p_copy[i][j+1]+p_copy[i][j-1])*rdy2
                                  -  rhs[i][j]
                              );
                    } else if (flag[i][j] & C_F) { 
                        /* modified star near boundary */
                        beta_mod = -omega/((eps_E+eps_W)*rdx2+(eps_N+eps_S)*rdy2);
                        p_copy[i][j] = (1.-omega)*p_copy[i][j] -
                            beta_mod*(
                                  (eps_E*p_copy[i+1][j]+eps_W*p_copy[i-1][j])*rdx2
                                + (eps_N*p_copy[i][j+1]+eps_S*p_copy[i][j-1])*rdy2
                                - rhs[i][j]
                            );
                    }
                } /* end of j */
            } /* end of i */
            communicate_1d(p_copy, imax, jmax, p_istart, p_iend, rank, size);
        } /* end of rb */


        // Check differences
        float total_error = 0.0;
        for (i = p_istart; i < p_iend; i++) {
            for (j = 0; j < jmax + 2; j++) {
                float err = p_copy[i][j] - p[i][j];
                total_error += err * err;
            }
        }
        printf("Total Square Error %f in process %d\n", total_error, rank);

        // Check all fails for rank 0
        if (rank == 0) {
            for (i = p_istart; i < p_iend; i++) {
                for (j = 0; j < jmax + 2; j++) {
                    float err = p_copy[i][j] - p[i][j];
                    if (err < 0.0001 && err > -0.0001) {continue;}
                    else {
                        printf("Break at i = %d, j = %d, off_by = %f\n", i, j, err);
                    }
                }
            }
        }

        
        /* Partial computation of residual */
        *res = 0.0;
        local_residual = 0.0;
        for (i = max(1, p_istart); i <= min(imax, p_iend - 1); i++) {
        //for (i = 1; i <= imax; i++) {
            for (j = 1; j <= jmax; j++) {
                if (flag[i][j] & C_F) {
                    /* only fluid cells */
                    add = (eps_E*(p[i+1][j]-p[i][j]) - 
                        eps_W*(p[i][j]-p[i-1][j])) * rdx2  +
                        (eps_N*(p[i][j+1]-p[i][j]) -
                        eps_S*(p[i][j]-p[i][j-1])) * rdy2  -  rhs[i][j];
                    local_residual += add*add;
                }
            }
        }
        MPI_Allreduce(&local_residual, res, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        *res = sqrt((*res)/ifull)/p0;

        /* convergence? */
        if (*res<eps) break;
    } /* end of iter */

    return iter;
}

/* Update the velocity values based on the tentative
 * velocity values and the new pressure matrix
 */
void update_velocity(float **u, float **v, float **f, float **g, float **p,
    char **flag, int imax, int jmax, float del_t, float delx, float dely)
{
    int i, j;

    for (i=1; i<=imax-1; i++) {
        for (j=1; j<=jmax; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                u[i][j] = f[i][j]-(p[i+1][j]-p[i][j])*del_t/delx;
            }
        }
    }
    for (i=1; i<=imax; i++) {
        for (j=1; j<=jmax-1; j++) {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
                v[i][j] = g[i][j]-(p[i][j+1]-p[i][j])*del_t/dely;
            }
        }
    }
}


/* Set the timestep size so that we satisfy the Courant-Friedrichs-Lewy
 * conditions (ie no particle moves more than one cell width in one
 * timestep). Otherwise the simulation becomes unstable.
 */
void set_timestep_interval(float *del_t, int imax, int jmax, float delx,
    float dely, float **u, float **v, float Re, float tau)
{
    int i, j;
    float umax, vmax, deltu, deltv, deltRe; 

    /* del_t satisfying CFL conditions */
    if (tau >= 1.0e-10) { /* else no time stepsize control */
        umax = 1.0e-10;
        vmax = 1.0e-10; 
        for (i=0; i<=imax+1; i++) {
            for (j=1; j<=jmax+1; j++) {
                umax = max(fabs(u[i][j]), umax);
            }
        }
        for (i=1; i<=imax+1; i++) {
            for (j=0; j<=jmax+1; j++) {
                vmax = max(fabs(v[i][j]), vmax);
            }
        }

        deltu = delx/umax;
        deltv = dely/vmax; 
        deltRe = 1/(1/(delx*delx)+1/(dely*dely))*Re/2.0;

        if (deltu<deltv) {
            *del_t = min(deltu, deltRe);
        } else {
            *del_t = min(deltv, deltRe);
        }
        *del_t = tau * (*del_t); /* multiply by safety factor */
    }
}
