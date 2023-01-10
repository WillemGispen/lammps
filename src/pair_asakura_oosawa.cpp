/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/*
Asakura-Oosawa pair potential (athermal)

\beta * u(r) = - A0 * (1 + A1 * r + A3 * r^3)    , 1 < r < (1+q) (attractive)
\beta * u(r) += WCA_{49,50} (r)                 , r < 50/49     (hard core)

where
A0 = \eta_p^r \frac{(1+q)^3}{q^3}
A1 = -\frac{3}{2*(1 + q)}
A3 = \frac{1}{2*(1 + q)^3}

Assumed: colloidal particles have diameter \sigma=1.
The coefficient passed via "pair_coeff" controls \eta_p^r,
 i.e. the packing fraction of the ideal polymer reservoir.

It also computes the derivative dU / d\eta_p^r, which is available via compute/pair.
*/

#include "pair_asakura_oosawa.h"

#include <cmath>
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"
#include <signal.h>



using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairAsakuraOosawa::PairAsakuraOosawa(LAMMPS *lmp) : Pair(lmp)
{
  // nextra = 1;
  // pvector = new double[1];
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairAsakuraOosawa::~PairAsakuraOosawa()
{
  delete [] pvector;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(rad);
    memory->destroy(cut);
    memory->destroy(a);
    memory->destroy(offset);
  }
}

/* ---------------------------------------------------------------------- */

void PairAsakuraOosawa::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r2inv,r,r3,rinv,screening,forceao,factor;
  double qp1, q, fac0, fac1, fac3;
  double fduds, duds;
  double r6inv, r12inv, r24inv, r48inv, b5049;
  int *ilist,*jlist,*numneigh,**firstneigh;

  T = 1.0;
  evdwl = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        r2inv = 1.0/rsq;
        r = sqrt(rsq);
        rinv = 1.0/r;
        r3 = rsq * r;
        qp1 = cut[itype][jtype];
        q = qp1 - 1.0;
        fac0 = a[itype][jtype] * qp1 * qp1 * qp1 / (q * q * q);
        fac1 = -3.0 / (2.0 * qp1);
        fac3 = 1.0 / (2.0 * qp1 * qp1 * qp1);
        forceao = T * fac0 * (fac1 + 3 * fac3 * rsq);

        if (r < 50.0/49.0) {
          // add continuous hard sphere approx WCA(50,49)
          r6inv = r2inv*r2inv*r2inv;
          r12inv = r6inv * r6inv;
          r24inv = r12inv * r12inv;
          r48inv = r24inv * r24inv;
          b5049 = 134.55266;
          forceao += T * 2.0 / 3.0 * b5049 * r48inv * r2inv * r2inv * (50.0 - 49.0 * r);
        }

        fpair = factor * forceao;

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          evdwl = - T * fac0 * (1.0 + fac1 * r + fac3 * r3) - offset[itype][jtype];

          if (eflag_global) {
            fduds = qp1 * qp1 * qp1 / (q * q * q);
            duds += - T * fduds * (1.0 + fac1 * r + fac3 * r3);
          }

          if (r < 50.0/49.0) {
            // add continuous hard sphere approx WCA(50,49)
            evdwl += T * 2.0 / 3.0 * b5049 * r48inv * (r2inv - rinv);
            evdwl += T * 2.0 / 3.0;
          }
          evdwl *= factor;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }

  // if (eflag_global) pvector[0] = duds;
  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairAsakuraOosawa::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(rad,n+1,"pair:rad");
  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(a,n+1,n+1,"pair:a");
  memory->create(offset,n+1,n+1,"pair:offset");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairAsakuraOosawa::settings(int narg, char **arg)
{
  if (narg != 1) error->all(FLERR,"Illegal pair_style command");

  cut_global = utils::numeric(FLERR,arg[0],false,lmp);
  // T = utils::numeric(FLERR,arg[1],false,lmp);  // temperature

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairAsakuraOosawa::coeff(int narg, char **arg)
{
  if (narg < 3 || narg > 4)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  double a_one = utils::numeric(FLERR,arg[2],false,lmp);

  double cut_one = cut_global;
  if (narg == 4) cut_one = utils::numeric(FLERR,arg[3],false,lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      a[i][j] = a_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairAsakuraOosawa::init_one(int i, int j)
{
  if (setflag[i][j] == 0) {
    a[i][j] = mix_energy(a[i][i],a[j][j],1.0,1.0);
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
  }

  if (offset_flag && (cut[i][j] > 0.0)) {
    error->all(FLERR,"Incorrect args for pair coefficients");
  } else offset[i][j] = 0.0;

  a[j][i] = a[i][j];
  offset[j][i] = offset[i][j];

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairAsakuraOosawa::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&a[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairAsakuraOosawa::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR,&setflag[i][j],sizeof(int),1,fp,nullptr,error);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR,&a[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&cut[i][j],sizeof(double),1,fp,nullptr,error);
        }
        MPI_Bcast(&a[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairAsakuraOosawa::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairAsakuraOosawa::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR,&cut_global,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&offset_flag,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&mix_flag,sizeof(int),1,fp,nullptr,error);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairAsakuraOosawa::write_data(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g\n",i,a[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairAsakuraOosawa::write_data_all(FILE *fp)
{
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g\n",i,j,a[i][j],cut[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairAsakuraOosawa::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                          double /*factor_coul*/, double factor_lj,
                          double &fforce)
{
  double r2inv,r,rinv,r3,screening,forceao,phi;
  double qp1, q, fac0, fac1, fac3;
  double r6inv, r12inv, r24inv, r48inv, b5049;

  T = 1.0;

  r2inv = 1.0/rsq;
  r = sqrt(rsq);
  r3 = rsq * r;
  rinv = 1.0/r;
  qp1 = cut[itype][jtype];
  q = qp1 - 1.0;
  fac0 = a[itype][jtype] * qp1 * qp1 * qp1 / (q * q * q);
  fac1 = -3.0 / (2.0 * qp1);
  fac3 = 1.0 / (2.0 * qp1 * qp1 * qp1);
  forceao = T * fac0 * (fac1 + 3.0 * fac3 * rsq);

  if (r < 50.0/49.0) {
    // add continuous hard sphere approx WCA(50,49)
    r6inv = r2inv*r2inv*r2inv;
    r12inv = r6inv * r6inv;
    r24inv = r12inv * r12inv;
    r48inv = r24inv * r24inv;
    b5049 = 134.55266;
    forceao += T * 2.0 / 3.0 * b5049 * r48inv * r2inv * r2inv * (50.0 - 49.0 * r);
  }

  fforce = factor_lj * forceao;

  phi = - T * fac0 * (1.0 + fac1 * r + fac3 * r3) - offset[itype][jtype];

  if (r < 50.0/49.0) {
    // add continuous hard sphere approx WCA(50,49)
    phi += T * 2.0 / 3.0 * b5049 * r48inv * (r2inv - rinv);
    phi += T * 2.0 / 3.0;
  }

  return factor_lj*phi;

}
