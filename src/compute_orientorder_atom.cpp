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

/* ----------------------------------------------------------------------
   Contributing author:  Aidan Thompson (SNL)
                         Axel Kohlmeyer (Temple U)
                         Willem Gispen (UU)
------------------------------------------------------------------------- */

#include "compute_orientorder_atom.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "math_special.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"

#include <cstring>
#include <cmath>

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace MathSpecial;


#ifdef DBL_EPSILON
  #define MY_EPSILON (10.0*DBL_EPSILON)
#else
  #define MY_EPSILON (10.0*2.220446049250313e-16)
#endif

#define QEPSILON 1.0e-6
#define ALLCOMP -21
#define AVERAGE_COMPONENTS 1
#define AVERAGE_INVARIANTS 2

/* ---------------------------------------------------------------------- */

ComputeOrientOrderAtom::ComputeOrientOrderAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  qlist(nullptr), distsq(nullptr), nearest(nullptr), rlist(nullptr),
  id_orientorder(nullptr), qn_local(nullptr), qn_neigh(nullptr),
  jjqlcomp_(nullptr), iql_(nullptr),
  qnarray(nullptr), qnm_r(nullptr), qnm_i(nullptr), cglist(nullptr)
{
  if (narg < 3 ) error->all(FLERR,"Illegal compute orientorder/atom command");

  // set default values for optional args

  nnn = 12;
  cutsq = 0.0;
  wlflag = 0;
  wlhatflag = 0;
  qlcompflag = 0;
  chunksize = 16384;

  // specify which orders to request

  nqlist = 5;
  memory->create(qlist,nqlist,"orientorder/atom:qlist");
  qlist[0] = 4;
  qlist[1] = 6;
  qlist[2] = 8;
  qlist[3] = 10;
  qlist[4] = 12;
  qmax = 12;

  // check for coarse-graining
  averageflag = 0;
  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"average") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute orientorder/atom command");
      if (strcmp(arg[iarg+1],"components") == 0) averageflag = AVERAGE_COMPONENTS;
      else if (strcmp(arg[iarg+1],"invariants") == 0) averageflag = AVERAGE_INVARIANTS;
      else if (strcmp(arg[iarg+1],"no") == 0) averageflag = 0;
      else error->all(FLERR,"Illegal compute orientorder/atom command");
      iarg += 2;
    }
    iarg += 1;
  }

  if (averageflag) {
    // read compute id, which should refer to another compute orientorder/atom
    iarg = 3;
    int iorientorder;
    while (iarg < narg) {
      if (strcmp(arg[iarg],"orientorder") == 0) {
          if (iarg+2 > narg)
            error->all(FLERR,"Illegal compute orientorder/atom command");

          int n = strlen(arg[iarg+1]) + 1;
          id_orientorder = new char[n];
          strcpy(id_orientorder,arg[iarg+1]);

          iorientorder = modify->find_compute(id_orientorder);
          c_orientorder = (ComputeOrientOrderAtom*)(modify->compute[iorientorder]);
          if (iorientorder < 0)
            error->all(FLERR,"Could not find compute orientorder/atom compute ID");
          if (!utils::strmatch(modify->compute[iorientorder]->style,"^orientorder/atom"))
            error->all(FLERR,"Compute orientorder/atom compute ID is not orientorder/atom");
          if (!(c_orientorder->qlcompflag) && (averageflag == AVERAGE_COMPONENTS))
            error->all(FLERR,"Compute orientorder/atom requires components "
                        "option in compute orientorder/atom");
          break;
      }
      iarg += 1;
    }

    // reset default values using the other orientorder compute
    nnn = c_orientorder->nnn;
    cutsq = c_orientorder->cutsq;
    wlflag = c_orientorder->wlflag;
    wlhatflag = c_orientorder->wlhatflag;
    commflag = 1;
    nqlist = c_orientorder->nqlist; 
    qlist = c_orientorder->qlist;
  }

  // process optional args

  iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"nnn") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute orientorder/atom command");
      if (strcmp(arg[iarg+1],"NULL") == 0) {
        nnn = 0;
      } else {
        nnn = utils::numeric(FLERR,arg[iarg+1],false,lmp);
        if (nnn <= 0)
          error->all(FLERR,"Illegal compute orientorder/atom command");
      }
      iarg += 2;
    } else if (strcmp(arg[iarg],"degrees") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute orientorder/atom command");
      nqlist = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (nqlist <= 0)
        error->all(FLERR,"Illegal compute orientorder/atom command");
      memory->destroy(qlist);
      memory->create(qlist,nqlist,"orientorder/atom:qlist");
      iarg += 2;
      if (iarg+nqlist > narg)
        error->all(FLERR,"Illegal compute orientorder/atom command");
      qmax = 0;
      for (int il = 0; il < nqlist; il++) {
        qlist[il] = utils::numeric(FLERR,arg[iarg+il],false,lmp);
        if (qlist[il] < 0)
          error->all(FLERR,"Illegal compute orientorder/atom command");
        if (qlist[il] > qmax) qmax = qlist[il];
      }
      iarg += nqlist;
    } else if (strcmp(arg[iarg],"wl") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute orientorder/atom command");
      if (strcmp(arg[iarg+1],"yes") == 0) wlflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) wlflag = 0;
      else error->all(FLERR,"Illegal compute orientorder/atom command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"wl/hat") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute orientorder/atom command");
      if (strcmp(arg[iarg+1],"yes") == 0) wlhatflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) wlhatflag = 0;
      else error->all(FLERR,"Illegal compute orientorder/atom command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"components") == 0) {
      qlcompflag = 1;
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute orientorder/atom command");
      if (strcmp(arg[iarg+1],"all") == 0) {
        qlcomp = ALLCOMP;
        iqlcomp = 0;
        break;
      } 
      qlcomp = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iqlcomp = -1;
      for (int il = 0; il < nqlist; il++)
        if (qlcomp == qlist[il]) {
          iqlcomp = il;
          break;
        }
      if (iqlcomp == -1)
        error->all(FLERR,"Illegal compute orientorder/atom command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"cutoff") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute orientorder/atom command");
      double cutoff = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (cutoff <= 0.0)
        error->all(FLERR,"Illegal compute orientorder/atom command");
      cutsq = cutoff*cutoff;
      iarg += 2;
    } else if (strcmp(arg[iarg],"chunksize") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute orientorder/atom command");
      chunksize = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (chunksize <= 0)
        error->all(FLERR,"Illegal compute orientorder/atom command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"comm") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute orientorder/atom command");
      if (strcmp(arg[iarg+1],"yes") == 0) commflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) commflag = 0;
      else error->all(FLERR,"Illegal compute orientorder/atom command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"average") == 0) {
      iarg += 2; 
    } else if (strcmp(arg[iarg],"orientorder") == 0) {
      iarg += 2; 
    } else error->all(FLERR,"Illegal compute orientorder/atom command");
  }

  ncol = nqlist;
  if (wlflag) ncol += nqlist;
  if (wlhatflag) ncol += nqlist;
  if (qlcompflag) {
    if (qlcomp == ALLCOMP) {
      for (int il = 0; il < nqlist; il++) {
        int l = qlist[il];
        ncol += 2*(2*l+1);
      }
    } else {
    ncol += 2*(2*qlcomp+1);
    }
  }

  if (averageflag) {
    // find index of other orientorder components
    int nqlist_ = c_orientorder->nqlist;
    jjqlcomp_0 = nqlist_; 
    if (c_orientorder->wlflag) jjqlcomp_0 += nqlist_;
    if (c_orientorder->wlhatflag) jjqlcomp_0 += nqlist_;

    // find indices of other orientorder ql and components
    // also compute size of qn_neigh
    ncol_qn_neigh = nqlist;
    memory->create(jjqlcomp_,nqlist,"orientorder/atom:jjqlcomp_");
    memory->create(iql_,jjqlcomp_0,"orientorder/atom:iql_");
    for (int il = 0; il < nqlist; il++) {
      int l = qlist[il];
      
      int found = 0;
      jjqlcomp_[il] = jjqlcomp_0;
      for (int il_ = 0; il_ < nqlist_; il_++) {
        int l_ = c_orientorder->qlist[il_];
        jjqlcomp_[il] += 2*(2*l_+1);
        if (l == l_) {
          iql_[il] = il_;
          ncol_qn_neigh += 2*(2*l+1);
          found = 1;
          break;
        }
      }
      if (!found) {
        error->all(FLERR,"Illegal compute orientorder/atom command");
      }
    }

    // find indices of third order invariants
    if (averageflag == AVERAGE_INVARIANTS) {
      int jj = nqlist;
      int jj_ = nqlist_;
      if (wlflag) {
        if (!c_orientorder->wlflag){
          error->all(FLERR,"Illegal compute orientorder/atom command");
        } else {
          for (int il = 0; il < nqlist; il++) {
            iql_[jj++] = iql_[jj_++];
          }
        }
      }
      if (wlhatflag) {
        if (!c_orientorder->wlhatflag){
          error->all(FLERR,"Illegal compute orientorder/atom command");
        } else {
          for (int il = 0; il < nqlist; il++) {
            iql_[jj++] = iql_[jj_++];
          }
        }
      }
    }
  }

  peratom_flag = 1;
  size_peratom_cols = ncol;

  nmax = 0;
  maxneigh = 0;
}

/* ---------------------------------------------------------------------- */

ComputeOrientOrderAtom::~ComputeOrientOrderAtom()
{
  if (copymode) return;

  memory->destroy(qnarray);
  memory->destroy(distsq);
  memory->destroy(rlist);
  memory->destroy(qn_neigh);
  memory->destroy(nearest);
  memory->destroy(qlist);
  memory->destroy(qnm_r);
  memory->destroy(qnm_i);
  memory->destroy(cglist);
}

/* ---------------------------------------------------------------------- */

void ComputeOrientOrderAtom::init()
{
  if (force->pair == nullptr)
    error->all(FLERR,"Compute orientorder/atom requires a "
               "pair style be defined");
  if (cutsq == 0.0) cutsq = force->pair->cutforce * force->pair->cutforce;
  else if (sqrt(cutsq) > force->pair->cutforce)
    error->all(FLERR,"Compute orientorder/atom cutoff is "
               "longer than pairwise cutoff");

  memory->create(qnm_r,nqlist,2*qmax+1,"orientorder/atom:qnm_r");
  memory->create(qnm_i,nqlist,2*qmax+1,"orientorder/atom:qnm_i");
  comm_forward = ncol_qn_neigh;

  // need an occasional full neighbor list

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 1;

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"orientorder/atom") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute orientorder/atom");

  if (wlflag || wlhatflag) init_clebsch_gordan();
}

/* ---------------------------------------------------------------------- */

void ComputeOrientOrderAtom::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeOrientOrderAtom::compute_peratom()
{
  int i,j,ii,jj,inum,jnum;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int *ilist,*jlist,*numneigh,**firstneigh;

  invoked_peratom = update->ntimestep;

  // grow order parameter array if necessary

  if (atom->nmax > nmax) {
    memory->destroy(qnarray);
    nmax = atom->nmax;
    memory->create(qnarray,nmax,ncol,"orientorder/atom:qnarray");
    array_atom = qnarray;
  }

  // invoke full neighbor list (will copy or build if necessary)

  neighbor->build_one(list);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // compute order parameter for each atom in group
  // use full neighbor list to count atoms less than cutoff

  double **x = atom->x;
  int *mask = atom->mask;
  memset(&qnarray[0][0],0,sizeof(double)*nmax*ncol);

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    double* qn = qnarray[i];
    if (mask[i] & groupbit) {
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      jlist = firstneigh[i];
      jnum = numneigh[i];

      // insure distsq and nearest arrays are long enough

      if (jnum > maxneigh) {
        memory->destroy(distsq);
        memory->destroy(rlist);
        memory->destroy(qn_neigh);
        memory->destroy(nearest);
        maxneigh = jnum+1;
        memory->create(distsq,maxneigh,"orientorder/atom:distsq");
        memory->create(rlist,maxneigh,3,"orientorder/atom:rlist");
        if (averageflag) {
          memory->create(qn_neigh,maxneigh,ncol_qn_neigh,"orientorder/atom:qn_neigh");
        }
        memory->create(nearest,maxneigh,"orientorder/atom:nearest");
      }

      // loop over list of all neighbors within force cutoff
      // distsq[] = distance sq to each
      // rlist[] = distance vector to each
      // nearest[] = atom indices of neighbors
      // qn_neigh[] = output vector of other compute_orientorder/atom of each neighbor and the particle itself

      int ncount = 0;
      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        j &= NEIGHMASK;

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;
        if (rsq < cutsq && (commflag || j < atom->nlocal)) {
          // if no communication, ignore ghost atoms
          distsq[ncount] = rsq;
          rlist[ncount][0] = delx;
          rlist[ncount][1] = dely;
          rlist[ncount][2] = delz;
          nearest[ncount++] = j;
        }
      }

      // if not nnn neighbors, order parameter = 0;

      if ((ncount == 0) || (ncount < nnn)) {
        for (int jj = 0; jj < ncol; jj++)
          qn[jj] = 0.0;
        continue;
      }

      // if nnn > 0, use only nearest nnn neighbors

      if (nnn > 0) {
        select3(nnn,ncount,distsq,nearest,rlist);
        ncount = nnn;
      }

      if (averageflag) {
        // compute and read orientorder of nearest neighbors and particle itself
        // compute
        if (!(c_orientorder->invoked_flag & INVOKED_PERATOM)) {
          c_orientorder->compute_peratom();
          c_orientorder->invoked_flag |= INVOKED_PERATOM;
        }
        if (commflag) {
          // TODO: fix seg fault here
          comm->forward_comm_compute(this);
        }
        nqlist = c_orientorder->nqlist;
        qn_local = c_orientorder->array_atom;

        // read
        ncount++;
        for (int jj = 0; jj < ncount; jj++) {
          if (jj == ncount-1) {
            nearest[jj] = i;
          }
          j = nearest[jj];
          
          for (int il = 0; il < nqlist; il++) {
            // read q values
            qn_neigh[jj][il] = qn_local[j][iql_[il]];
            if (averageflag == AVERAGE_COMPONENTS) {
              // read q components
              int l = qlist[il];
              for (int k = 0; k < 2*(2*l+1); k++){
                qn_neigh[jj][nqlist + k] = qn_local[j][jjqlcomp_[il] + k];
              }
            }
          }
        }        
      }

      calc_boop(rlist, qn_neigh, ncount, qn, qlist, nqlist);
    }
  }
}


/* ---------------------------------------------------------------------- */

int ComputeOrientOrderAtom::pack_forward_comm(int n, int *list, double *buf,
                                        int /*pbc_flag*/, int * /*pbc*/)
{ 
  int i,m=0,j;
  for (i = 0; i < n; ++i) {
    for (int il = 0; il < nqlist; il++) {
      buf[m++] = qn_local[i][il];
      if (averageflag == AVERAGE_COMPONENTS) {
        int l = qlist[il];
        for (int k = 0; k < 2*(2*l+1); k++){
          buf[m++] = qn_local[i][jjqlcomp_[il] + k];
        }
      }
    }

  }

  return m;
}

/* ---------------------------------------------------------------------- */

void ComputeOrientOrderAtom::unpack_forward_comm(int n, int first, double *buf)
{
  int i,last,m=0,j;
  last = first + n;
  for (i = first; i < last; ++i) {
    for (int il = 0; il < nqlist; il++) {
      qn_local[i][il] = buf[m++];
      if (averageflag == AVERAGE_COMPONENTS) {
        int l = qlist[il];
        for (int k = 0; k < 2*(2*l+1); k++){
          qn_local[i][jjqlcomp_[il] + k] = buf[m++];
        }
      }
    }

  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeOrientOrderAtom::memory_usage()
{
  double bytes = (double)ncol*nmax * sizeof(double);
  bytes += (double)(qmax*(2*qmax+1)+maxneigh*4) * sizeof(double);
  bytes += (double)(nqlist+maxneigh) * sizeof(int);
  return bytes;
}

/* ----------------------------------------------------------------------
   select3 routine from Numerical Recipes (slightly modified)
   find k smallest values in array of length n
   sort auxiliary arrays at same time
------------------------------------------------------------------------- */

// Use no-op do while to create single statement

#define SWAP(a,b) do {       \
    tmp = a; a = b; b = tmp; \
  } while (0)

#define ISWAP(a,b) do {        \
    itmp = a; a = b; b = itmp; \
  } while (0)

#define SWAP3(a,b) do {                  \
    tmp = a[0]; a[0] = b[0]; b[0] = tmp; \
    tmp = a[1]; a[1] = b[1]; b[1] = tmp; \
    tmp = a[2]; a[2] = b[2]; b[2] = tmp; \
  } while (0)

/* ---------------------------------------------------------------------- */

void ComputeOrientOrderAtom::select3(int k, int n, double *arr, int *iarr, double **arr3)
{
  int i,ir,j,l,mid,ia,itmp;
  double a,tmp,a3[3];

  arr--;
  iarr--;
  arr3--;
  l = 1;
  ir = n;
  for (;;) {
    if (ir <= l+1) {
      if (ir == l+1 && arr[ir] < arr[l]) {
        SWAP(arr[l],arr[ir]);
        ISWAP(iarr[l],iarr[ir]);
        SWAP3(arr3[l],arr3[ir]);
      }
      return;
    } else {
      mid=(l+ir) >> 1;
      SWAP(arr[mid],arr[l+1]);
      ISWAP(iarr[mid],iarr[l+1]);
      SWAP3(arr3[mid],arr3[l+1]);
      if (arr[l] > arr[ir]) {
        SWAP(arr[l],arr[ir]);
        ISWAP(iarr[l],iarr[ir]);
        SWAP3(arr3[l],arr3[ir]);
      }
      if (arr[l+1] > arr[ir]) {
        SWAP(arr[l+1],arr[ir]);
        ISWAP(iarr[l+1],iarr[ir]);
        SWAP3(arr3[l+1],arr3[ir]);
      }
      if (arr[l] > arr[l+1]) {
        SWAP(arr[l],arr[l+1]);
        ISWAP(iarr[l],iarr[l+1]);
        SWAP3(arr3[l],arr3[l+1]);
      }
      i = l+1;
      j = ir;
      a = arr[l+1];
      ia = iarr[l+1];
      a3[0] = arr3[l+1][0];
      a3[1] = arr3[l+1][1];
      a3[2] = arr3[l+1][2];
      for (;;) {
        do i++; while (arr[i] < a);
        do j--; while (arr[j] > a);
        if (j < i) break;
        SWAP(arr[i],arr[j]);
        ISWAP(iarr[i],iarr[j]);
        SWAP3(arr3[i],arr3[j]);
      }
      arr[l+1] = arr[j];
      arr[j] = a;
      iarr[l+1] = iarr[j];
      iarr[j] = ia;
      arr3[l+1][0] = arr3[j][0];
      arr3[l+1][1] = arr3[j][1];
      arr3[l+1][2] = arr3[j][2];
      arr3[j][0] = a3[0];
      arr3[j][1] = a3[1];
      arr3[j][2] = a3[2];
      if (j >= k) ir = j-1;
      if (j <= k) l = i;
    }
  }
}

/* ----------------------------------------------------------------------
   calculate the bond orientational order parameters
------------------------------------------------------------------------- */

void ComputeOrientOrderAtom::calc_boop(double **rlist,
                                       double **qn_neigh,
                                       int ncount, double qn[],
                                       int qlist[], int nqlist) {

  if (averageflag == AVERAGE_INVARIANTS) {
    // average orientorder of neighbor
    for (int ineigh = 0; ineigh < ncount; ineigh++) {
      const double * const qn_ = qn_neigh[ineigh];

      for (int il = 0; il < ncol; il++) {
        // TODO: fix average over wl giving zero/nan
        int il_ = iql_[il];
        double q = qn_[il_];
        qn[il] += q / ncount;
      }
    }
    return;
  }

  for (int il = 0; il < nqlist; il++) {
    int l = qlist[il];
    for (int m = 0; m < 2*l+1; m++) {
      qnm_r[il][m] = 0.0;
      qnm_i[il][m] = 0.0;
    }
  }

  for (int ineigh = 0; ineigh < ncount; ineigh++) {
      const double * const r = rlist[ineigh];
      double rmag = dist(r);
      if (rmag <= MY_EPSILON) {
        return;
      }

    if (averageflag == AVERAGE_COMPONENTS) {
      // get orientorder components of neighbor
      // TODO: fix zero/nans
      const double * const qn_ = qn_neigh[ineigh];

      for (int il = 0; il < nqlist; il++) {
        int jj = jjqlcomp_[il];
        int l = qlist[il];
        double qnormfac = sqrt(MY_4PI/(2*l+1));
        int il_ = iql_[il];
        double qnfac = qn_[il_]/qnormfac;

        // calculate sum of orientorder components over neighbors
        for (int m = 0; m < 2*l+1; m++) {
          double qnm_r_lm_ = qn_[jj++] * qnfac;
          double qnm_i_lm_ = qn_[jj++] * qnfac;
          qnm_r[il][m] += qnm_r_lm_;
          qnm_i[il][m] += qnm_i_lm_;
        }
      }
    } else {
      double costheta = r[2] / rmag;
      double expphi_r = r[0];
      double expphi_i = r[1];
      double rxymag = sqrt(expphi_r*expphi_r+expphi_i*expphi_i);
      if (rxymag <= MY_EPSILON) {
        expphi_r = 1.0;
        expphi_i = 0.0;
      } else {
        double rxymaginv = 1.0/rxymag;
        expphi_r *= rxymaginv;
        expphi_i *= rxymaginv;
      }

      for (int il = 0; il < nqlist; il++) {
        int l = qlist[il];

        // calculate spherical harmonics
        // Ylm, -l <= m <= l
        // sign convention: sign(Yll(0,0)) = (-1)^l

        qnm_r[il][l] += polar_prefactor(l, 0, costheta);
        double expphim_r = expphi_r;
        double expphim_i = expphi_i;
        for (int m = 1; m <= +l; m++) {

          double prefactor = polar_prefactor(l, m, costheta);
          double ylm_r = prefactor * expphim_r;
          double ylm_i = prefactor * expphim_i;
          qnm_r[il][m+l] += ylm_r;
          qnm_i[il][m+l] += ylm_i;
          if (m & 1) {
            qnm_r[il][-m+l] -= ylm_r;
            qnm_i[il][-m+l] += ylm_i;
          } else {
            qnm_r[il][-m+l] += ylm_r;
            qnm_i[il][-m+l] -= ylm_i;
          }
          double tmp_r = expphim_r*expphi_r - expphim_i*expphi_i;
          double tmp_i = expphim_r*expphi_i + expphim_i*expphi_r;
          expphim_r = tmp_r;
          expphim_i = tmp_i;
        }

      }
    }
  }

  // convert sums to averages

  double facn = 1.0 / ncount;
  for (int il = 0; il < nqlist; il++) {
    int l = qlist[il];
    for (int m = 0; m < 2*l+1; m++) {
      qnm_r[il][m] *= facn;
      qnm_i[il][m] *= facn;
    }
  }

  // calculate Q_l
  // NOTE: optional W_l_hat and components of Q_qlcomp use these stored Q_l values

  int jj = 0;
  for (int il = 0; il < nqlist; il++) {
    int l = qlist[il];
    double qnormfac = sqrt(MY_4PI/(2*l+1));
    double qm_sum = 0.0;
    for (int m = 0; m < 2*l+1; m++)
      qm_sum += qnm_r[il][m]*qnm_r[il][m] + qnm_i[il][m]*qnm_i[il][m];
    qn[jj++] = qnormfac * sqrt(qm_sum);
  }

  // calculate W_l

  if (wlflag) {
    int idxcg_count = 0;
    for (int il = 0; il < nqlist; il++) {
      int l = qlist[il];
      double wlsum = 0.0;
      for (int m1 = 0; m1 < 2*l+1; m1++) {
        for (int m2 = MAX(0,l-m1); m2 < MIN(2*l+1,3*l-m1+1); m2++) {
          int m = m1 + m2 - l;
          double qm1qm2_r = qnm_r[il][m1]*qnm_r[il][m2] - qnm_i[il][m1]*qnm_i[il][m2];
          double qm1qm2_i = qnm_r[il][m1]*qnm_i[il][m2] + qnm_i[il][m1]*qnm_r[il][m2];
          wlsum += (qm1qm2_r*qnm_r[il][m] + qm1qm2_i*qnm_i[il][m])*cglist[idxcg_count];
          idxcg_count++;
        }
      }
      qn[jj++] = wlsum/sqrt(2*l+1);
    }
  }

  // calculate W_l_hat

  if (wlhatflag) {
    int idxcg_count = 0;
    for (int il = 0; il < nqlist; il++) {
      int l = qlist[il];
      double wlsum = 0.0;
      for (int m1 = 0; m1 < 2*l+1; m1++) {
        for (int m2 = MAX(0,l-m1); m2 < MIN(2*l+1,3*l-m1+1); m2++) {
          int m = m1 + m2 - l;
          double qm1qm2_r = qnm_r[il][m1]*qnm_r[il][m2] - qnm_i[il][m1]*qnm_i[il][m2];
          double qm1qm2_i = qnm_r[il][m1]*qnm_i[il][m2] + qnm_i[il][m1]*qnm_r[il][m2];
          wlsum += (qm1qm2_r*qnm_r[il][m] + qm1qm2_i*qnm_i[il][m])*cglist[idxcg_count];
          idxcg_count++;
        }
      }
      if (qn[il] < QEPSILON)
        qn[jj++] = 0.0;
      else {
        double qnormfac = sqrt(MY_4PI/(2*l+1));
        double qnfac = qnormfac/qn[il];
        qn[jj++] = wlsum/sqrt(2*l+1)*(qnfac*qnfac*qnfac);
      }
    }
  }

  // Calculate components of Q_l/|Q_l|, for l=qlcomp
  // or for all l if qlcomp == ALLCOMP

  if (qlcompflag) {
    int ilstart, ilstop;
    if (qlcomp == ALLCOMP) {
      ilstart = 0;
      ilstop = nqlist;
    } else {
      ilstart = iqlcomp;
      ilstop = ilstart + 1;
    }

    for (int il = ilstart; il < ilstop; il++) {
      int l = qlist[il];
      if (qn[il] < QEPSILON)
        for (int m = 0; m < 2*l+1; m++) {
          qn[jj++] = 0.0;
          qn[jj++] = 0.0;
        }
      else {
        double qnormfac = sqrt(MY_4PI/(2*l+1));
        double qnfac = qnormfac/qn[il];
        for (int m = 0; m < 2*l+1; m++) {
          qn[jj++] = qnm_r[il][m] * qnfac;
          qn[jj++] = qnm_i[il][m] * qnfac;
      }
      }
    }
  }
}

/* ----------------------------------------------------------------------
   calculate scalar distance
------------------------------------------------------------------------- */

double ComputeOrientOrderAtom::dist(const double r[])
{
  return sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
}

/* ----------------------------------------------------------------------
   polar prefactor for spherical harmonic Y_l^m, where
   Y_l^m (theta, phi) = prefactor(l, m, cos(theta)) * exp(i*m*phi)
------------------------------------------------------------------------- */

double ComputeOrientOrderAtom::polar_prefactor(int l, int m, double costheta)
{
  const int mabs = abs(m);

  double prefactor = 1.0;
  for (int i=l-mabs+1; i < l+mabs+1; ++i)
    prefactor *= static_cast<double>(i);

  prefactor = sqrt(static_cast<double>(2*l+1)/(MY_4PI*prefactor))
    * associated_legendre(l,mabs,costheta);

  if ((m < 0) && (m % 2)) prefactor = -prefactor;

  return prefactor;
}

/* ----------------------------------------------------------------------
   associated legendre polynomial
   sign convention: P(l,l) = (2l-1)!!(-sqrt(1-x^2))^l
------------------------------------------------------------------------- */

double ComputeOrientOrderAtom::associated_legendre(int l, int m, double x)
{
  if (l < m) return 0.0;

  double p(1.0), pm1(0.0), pm2(0.0);

  if (m != 0) {
    const double msqx = -sqrt(1.0-x*x);
    for (int i=1; i < m+1; ++i)
      p *= static_cast<double>(2*i-1) * msqx;
  }

  for (int i=m+1; i < l+1; ++i) {
    pm2 = pm1;
    pm1 = p;
    p = (static_cast<double>(2*i-1)*x*pm1
         - static_cast<double>(i+m-1)*pm2) / static_cast<double>(i-m);
  }

  return p;
}

/* ----------------------------------------------------------------------
   assign Clebsch-Gordan coefficients
   using the quasi-binomial formula VMK 8.2.1(3)
   specialized for case j1=j2=j=l
------------------------------------------------------------------------- */

void ComputeOrientOrderAtom::init_clebsch_gordan()
{
  double sum,dcg,sfaccg, sfac1, sfac2;
  int m, aa2, bb2, cc2;
  int ifac, idxcg_count;

  idxcg_count = 0;
  for (int il = 0; il < nqlist; il++) {
    int l = qlist[il];
    for (int m1 = 0; m1 < 2*l+1; m1++)
      for (int m2 = MAX(0,l-m1); m2 < MIN(2*l+1,3*l-m1+1); m2++)
        idxcg_count++;
  }
  idxcg_max = idxcg_count;
  memory->create(cglist, idxcg_max, "computeorientorderatom:cglist");

  idxcg_count = 0;
  for (int il = 0; il < nqlist; il++) {
    int l = qlist[il];
    for (int m1 = 0; m1 < 2*l+1; m1++) {
        aa2 = m1 - l;
        for (int m2 = MAX(0,l-m1); m2 < MIN(2*l+1,3*l-m1+1); m2++) {
          bb2 = m2 - l;
          m = aa2 + bb2 + l;

          sum = 0.0;
          for (int z = MAX(0, MAX(-aa2, bb2));
               z <= MIN(l, MIN(l - aa2, l + bb2)); z++) {
            ifac = z % 2 ? -1 : 1;
            sum += ifac /
              (factorial(z) *
               factorial(l - z) *
               factorial(l - aa2 - z) *
               factorial(l + bb2 - z) *
               factorial(aa2 + z) *
               factorial(-bb2 + z));
          }

          cc2 = m - l;
          sfaccg = sqrt(factorial(l + aa2) *
                        factorial(l - aa2) *
                        factorial(l + bb2) *
                        factorial(l - bb2) *
                        factorial(l + cc2) *
                        factorial(l - cc2) *
                        (2*l + 1));

          sfac1 = factorial(3*l + 1);
          sfac2 = factorial(l);
          dcg = sqrt(sfac2*sfac2*sfac2 / sfac1);

          cglist[idxcg_count] = sum * dcg * sfaccg;
          idxcg_count++;
        }
      }
  }
}
