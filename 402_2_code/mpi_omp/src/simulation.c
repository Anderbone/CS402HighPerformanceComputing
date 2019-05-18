#include <mpi.h>
#include <omp.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "datadef.h"
#include "init.h"



#define max(x,y) ((x)>(y)?(x):(y))
#define min(x,y) ((x)<(y)?(x):(y))

extern int nprocs, proc;
extern int ileft, iright, each_part;

MPI_Status status;

/* Computation of tentative velocity field (f, g) */
void compute_tentative_velocity(float **u, float **v, float **f, float **g,
    char **flag, int imax, int jmax, float del_t, float delx, float dely,
    float gamma, float Re)
{
    int  i, j;
    float du2dx, duvdy, duvdx, dv2dy, laplu, laplv;

    // for (i=1; i<=imax-1; i++) {
    //chagne, add

    // only the last part need to compute i -1
    int iright_temp = iright;

    if (proc == nprocs -1){
        iright_temp -= 1;
    }

    // #pragma omp parallel for schedule(static) private(i,j)
    // #pragma omp parallel for schedule(static) private(i, j, du2dx,duvdy,laplu)
      #pragma omp parallel for schedule(static) private(du2dx,duvdy,laplu)


    for (i=ileft; i<=iright_temp; i++) {
        for (j=1; j<=jmax; j++) {
            /* only if both adjacent cells are fluid cells */

            // printf("total threads %d\n", omp_get_num_threads());
            // printf("threads %d\n", omp_get_thread_num());
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

    #pragma omp parallel for schedule(static) private(duvdx,dv2dy,laplv)

    // for (i=1; i<=imax; i++) {
    for (i=ileft; i<=iright; i++) {
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
    // #pragma omp parallel for schedule(static) private(j)
    for (j=1; j<=jmax; j++) {
        f[0][j]    = u[0][j];
        f[imax][j] = u[imax][j];
    }
    // for (i=1; i<=imax; i++) {
    // #pragma omp parallel for schedule(static) private(i)
    for (i=ileft; i<=iright; i++) {
        g[i][0]    = v[i][0];
        g[i][jmax] = v[i][jmax];
    }

    // MPI_Barrier(MPI_COMM_WORLD);

    if (proc != nprocs - 1) {
    // Pass to the right
        MPI_Send(&f[iright][0], jmax+2, MPI_FLOAT, proc+1, 0, MPI_COMM_WORLD);
        MPI_Send(&g[iright][0], jmax+2, MPI_FLOAT, proc+1, 0, MPI_COMM_WORLD);

        MPI_Recv(&f[iright+1][0], jmax+2, MPI_FLOAT, proc+1, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&g[iright+1][0], jmax+2, MPI_FLOAT, proc+1, 0, MPI_COMM_WORLD, &status);
    } if (proc != 0) {
    // pass to the left
        MPI_Recv(&f[ileft-1][0], jmax+2, MPI_FLOAT, proc-1, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&g[ileft-1][0], jmax+2, MPI_FLOAT, proc-1, 0, MPI_COMM_WORLD, &status);

        MPI_Send(&f[ileft][0], jmax+2, MPI_FLOAT, proc-1, 0, MPI_COMM_WORLD);
        MPI_Send(&g[ileft][0], jmax+2, MPI_FLOAT, proc-1, 0, MPI_COMM_WORLD);
    }
    // MPI_Allgather(MPI_IN_PLACE, each_part*(jmax+2), MPI_FLOAT, &f[1][0], each_part*(jmax+2), MPI_FLOAT, MPI_COMM_WORLD);
    // MPI_Allgather(MPI_IN_PLACE, each_part*(jmax+2), MPI_FLOAT, &g[1][0], each_part*(jmax+2), MPI_FLOAT, MPI_COMM_WORLD);
}


/* Calculate the right hand side of the pressure equation */
void compute_rhs(float **f, float **g, float **rhs, char **flag, int imax,
    int jmax, float del_t, float delx, float dely)
{
    int i, j;

    // #pragma omp parallel for schedule(static) private(i,j)

    for (i=ileft;i<=iright;i++) {
    // for (i=1;i<=imax;i++) {
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
/* Red/Black SOR to solve the poisson equation */
int poisson(float **p, float **rhs, char **flag, int imax, int jmax,
    float delx, float dely, float eps, int itermax, float omega,
    float *res, int ifull)
{
    int i, j, iter;
    float add, beta_2, beta_mod;
    float p0 = 0.0;

    int rb; /* Red-black value. */

    float rdx2 = 1.0/(delx*delx);
    float rdy2 = 1.0/(dely*dely);
    beta_2 = -omega/(2.0*(rdx2+rdy2));

    // add
    float p0_all = 0.0;   // used to reduction later

    /* Calculate sum of squares */
    // #pragma omp parallel for schedule(static) private(i,j) reduction(+:p0)
    for (i = ileft; i <= iright; i++) {
        for (j=1; j<=jmax; j++) {
            if (flag[i][j] & C_F) { p0 += p[i][j]*p[i][j]; }
        }
    }

    //Reduce p0 from all parts to correctly get p0_all, and put it to each part
    MPI_Allreduce(&p0, &p0_all, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    // p0 = sqrt(p0/ifull);
    p0 = sqrt(p0_all/ifull);

    if (p0 < 0.0001) { p0 = 1.0; }

    /* Red/Black SOR-iteration */
    for (iter = 0; iter < itermax; iter++) {
        for (rb = 0; rb <= 1; rb++) {
          //change  here
          	#pragma omp parallel for schedule(static) private(beta_mod)
            for (i = ileft; i <= iright; i++) {
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

          // add here
        // MPI_Barrier(MPI_COMM_WORLD);
        if (proc != nprocs - 1) {
      // Pass only to the right
          MPI_Send(&p[iright][0], jmax+2, MPI_FLOAT, proc+1, 0, MPI_COMM_WORLD);
          MPI_Recv(&p[iright+1][0], jmax+2, MPI_FLOAT, proc+1, 0, MPI_COMM_WORLD, &status);
        } if (proc != 0) {
      // pass only to the left
          MPI_Recv(&p[ileft-1][0], jmax+2, MPI_FLOAT, proc-1, 0, MPI_COMM_WORLD, &status);
          MPI_Send(&p[ileft][0], jmax+2, MPI_FLOAT, proc-1, 0, MPI_COMM_WORLD);
      }
        } /* end of rb */



        /* Partial computation of residual */
        //add
        *res = 0.0;
        float add2 = 0.0; //  add*add in each processor
        float add2_all = 0.0; // total
        // #pragma omp parallel for schedule(static) private(i,j, add) reduction(+:add2)
        for (i = ileft; i <= iright; i++) {
          // for (i = 1; i <= imax; i++) {
            for (j = 1; j <= jmax; j++) {
                if (flag[i][j] & C_F) {
                    /* only fluid cells */
                    add = (eps_E*(p[i+1][j]-p[i][j]) -
                        eps_W*(p[i][j]-p[i-1][j])) * rdx2  +
                        (eps_N*(p[i][j+1]-p[i][j]) -
                        eps_S*(p[i][j]-p[i][j-1])) * rdy2  -  rhs[i][j];
                    // *res += add*add;
                    //add
                    add2 += add*add;
                }
            }
        }
        MPI_Allreduce(&add2, &add2_all, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        // *res = sqrt((*res)/ifull)/p0;
        *res = sqrt((add2_all)/ifull)/p0;

        /* convergence? */
        if (*res<eps) break;
    } /* end of iter */

    // if (proc == 0) {
    //     MPI_Gather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &p[1][0], each_part*(jmax+2), MPI_FLOAT, 0, MPI_COMM_WORLD);
    // } else {
    //     MPI_Gather(&p[ileft][0], each_part*(jmax+2), MPI_FLOAT, NULL, each_part*(jmax+2), MPI_FLOAT,  0, MPI_COMM_WORLD);
    // }
    // MPI_Bcast(&p[0][0],(imax+2)*(jmax+2), MPI_FLOAT,0,MPI_COMM_WORLD);
    // MPI_Allgather(&p[ileft_temp][0], each_part*(jmax+2), MPI_FLOAT, &p[0][0], each_part*(jmax+2), MPI_FLOAT, MPI_COMM_WORLD);
    return iter;
}


/* Update the velocity values based on the tentative
 * velocity values and the new pressure matrix
 */
 void update_velocity(float **u, float **v, float **f, float **g, float **p,
     char **flag,  int imax, int jmax, float del_t, float delx, float dely)
 {
     int i, j;


     // for (i=1; i<=imax-1; i++) {
     // If we're in the right most chunk, ignore last column
     // cf. original for statement had: i=1; i<imax-1; i++
     int iright_temp = iright;
     if (iright_temp == imax) {
         iright_temp -= 1;
     }
     // #pragma omp parallel for schedule(static) private(i,j)
     for (i=ileft; i<=iright_temp; i++) {
         for (j=1; j<=jmax; j++) {
             /* only if both adjacent cells are fluid cells */
             if ((flag[i][j] & C_F) && (flag[i+1][j] & C_F)) {
                 u[i][j] = f[i][j]-(p[i+1][j]-p[i][j])*del_t/delx;
             }
         }
     }
       // #pragma omp parallel for schedule(static) private(i,j)
     for (i=ileft; i<=iright; i++) {
         for (j=1; j<=jmax-1; j++) {
             /* only if both adjacent cells are fluid cells */
             if ((flag[i][j] & C_F) && (flag[i][j+1] & C_F)) {
                 v[i][j] = g[i][j]-(p[i][j+1]-p[i][j])*del_t/dely;
             }
         }
     }

     // MPI_Barrier(MPI_COMM_WORLD);
     if (proc != nprocs - 1) {
     // Pass only to the right
         MPI_Send(&u[iright][0], jmax+2, MPI_FLOAT, proc+1, 0, MPI_COMM_WORLD);
         MPI_Recv(&u[iright+1][0], jmax+2, MPI_FLOAT, proc+1, 0, MPI_COMM_WORLD, &status);
       } if (proc != 0) {
     // pass only to the left
         MPI_Recv(&u[ileft-1][0], jmax+2, MPI_FLOAT, proc-1, 0, MPI_COMM_WORLD, &status);
         MPI_Send(&u[ileft][0], jmax+2, MPI_FLOAT, proc-1, 0, MPI_COMM_WORLD);
     }

     // MPI_Barrier(MPI_COMM_WORLD);
     MPI_Allgather(MPI_IN_PLACE, each_part*(jmax+2), MPI_FLOAT, &v[1][0], each_part*(jmax+2), MPI_FLOAT, MPI_COMM_WORLD);
     MPI_Allgather(MPI_IN_PLACE, each_part*(jmax+2), MPI_FLOAT, &u[1][0], each_part*(jmax+2), MPI_FLOAT, MPI_COMM_WORLD);
     MPI_Allgather(MPI_IN_PLACE, each_part*(jmax+2), MPI_FLOAT, &p[1][0], each_part*(jmax+2), MPI_FLOAT, MPI_COMM_WORLD);


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

     float umax_all, vmax_all;

     /* del_t satisfying CFL conditions */
     if (tau >= 1.0e-10) { /* else no time stepsize control */
         umax = 1.0e-10;
         vmax = 1.0e-10;

         //remember to use temp here,
         //or it would change ileft and iright in other places and make wrong results.
         int ileft_temp;
         int iright_temp;

         ileft_temp = ileft;
         if (proc == 0){
           ileft_temp = 0;
         }


         iright_temp = iright;
         if (proc == nprocs -1){
           iright_temp = imax+1;
         }


         // for (i=0; i<=imax+1; i++) {
         // #pragma omp parallel for schedule(static) private(i,j) reduction(max:umax)

         for (i=ileft_temp; i<=iright_temp; i++) {
             for (j=1; j<=jmax+1; j++) {
                 umax = max(fabs(u[i][j]), umax);
             }
         }
         // for (i=1; i<=imax+1; i++) {
            // #pragma omp parallel for schedule(static) private(i,j) reduction(max:vmax)
           for (i=ileft_temp; i<=iright_temp; i++) {
             for (j=0; j<=jmax+1; j++) {
                 vmax = max(fabs(v[i][j]), vmax);
             }
         }

         MPI_Allreduce(&umax, &umax_all, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
         MPI_Allreduce(&vmax, &vmax_all, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

         // deltu = delx/umax;
         // deltv = dely/vmax;
         deltu = delx/umax_all;
         deltv = dely/vmax_all;
         deltRe = 1/(1/(delx*delx)+1/(dely*dely))*Re/2.0;

         if (deltu<deltv) {
             *del_t = min(deltu, deltRe);
         } else {
             *del_t = min(deltv, deltRe);
         }
         *del_t = tau * (*del_t); /* multiply by safety factor */
     }
 }
