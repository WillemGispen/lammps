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
                         Koenraad Janssens and David Olmsted (SNL)
                         Willem Gispen (UU)
------------------------------------------------------------------------- */

#include "compute_angle_atom.h"
#include "VORONOI/compute_voronoi_atom.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "math_eigen.h"
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

#ifdef DBL_EPSILON
  #define MY_EPSILON (10.0*DBL_EPSILON)
#else
  #define MY_EPSILON (10.0*2.220446049250313e-16)
#endif

#define QEPSILON 1.0e-6

#define ALLCOMP -21
#define SANN -16
#define VORO -17
#define INVOKED_PERATOM 8
#define INVOKED_LOCAL 16


/* ---------------------------------------------------------------------- */

ComputeAngleAtom::ComputeAngleAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  distsq(nullptr), nearest(nullptr), rlist(nullptr),
  sort(nullptr), id_voronoi(nullptr), voro_local(nullptr), angle_array(nullptr)
{
  if (narg < 3 ) error->all(FLERR,"Illegal compute angle/atom command");

  // set default values for optional args

  nnn = 12;
  bins = 0;
  cutsq = 0.0;
  chunksize = 16384;
  nqlist = 0;

  // process optional args

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"nnn") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute angle/atom command");
      if (strcmp(arg[iarg+1],"NULL") == 0) {
        nnn = 0;
      } else if (strcmp(arg[iarg+1],"SANN") == 0) {
        nnn = SANN;
      } else if (strcmp(arg[iarg+1],"VORO") == 0) {
        nnn = VORO;
      } else {
        nnn = utils::numeric(FLERR,arg[iarg+1],false,lmp);
        if (nnn <= 0)
          error->all(FLERR,"Illegal compute angle/atom command");
      }
      iarg += 2;
    } else if (strcmp(arg[iarg],"bins") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute angle/atom command");
      if (strcmp(arg[iarg+1],"NULL") == 0) {
        bins = 0;
      } else {
        bins = utils::numeric(FLERR,arg[iarg+1],false,lmp);
        if (bins <= 0)
          error->all(FLERR,"Illegal compute angle/atom command");
      }
      iarg += 2;
    } else if (strcmp(arg[iarg],"voronoi") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute angle/atom command");

      int n = strlen(arg[iarg+1]) + 1;
      id_voronoi = new char[n];
      strcpy(id_voronoi,arg[iarg+1]);

      int ivoronoi = modify->find_compute(id_voronoi);
      if (ivoronoi < 0)
        error->all(FLERR,"Could not find compute voronoi/atom compute ID");
      if (!utils::strmatch(modify->compute[ivoronoi]->style,"^voronoi/atom"))
        error->all(FLERR,"Compute angle/atom compute ID is not voronoi/atom");
      c_voronoi = (ComputeVoronoi*)(modify->compute[ivoronoi]);
      if (c_voronoi->faces_flag != 2) {
        error->all(FLERR,"Compute angle/atom: voronoi compute should have neighbors yes_local_id");
      }
      
      iarg += 2;
    } else if (strcmp(arg[iarg],"cutoff") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute angle/atom command");
      double cutoff = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (cutoff <= 0.0)
        error->all(FLERR,"Illegal compute angle/atom command");
      cutsq = cutoff*cutoff;
      iarg += 2;
    } else if (strcmp(arg[iarg],"chunksize") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute angle/atom command");
      chunksize = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (chunksize <= 0)
        error->all(FLERR,"Illegal compute angle/atom command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"degrees") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal ompute angle/atom command");
      nqlist = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (nqlist <= 0)
        error->all(FLERR,"Illegal compute angle/atom command");
      memory->create(qlist,nqlist,"angle/atom:qlist");
      iarg += 2;
      if (iarg+nqlist > narg)
        error->all(FLERR,"Illegal compute angle/atom command");
      // qmax = 0;
      for (int il = 0; il < nqlist; il++) {
        qlist[il] = utils::numeric(FLERR,arg[iarg+il],false,lmp);
        if (qlist[il] < 0)
          error->all(FLERR,"Illegal compute angle/atom command");
        // if (qlist[il] > qmax) qmax = qlist[il];
      }
      iarg += nqlist;
    } else error->all(FLERR,"Illegal compute angle/atom command");
  }

  peratom_flag = 1;
  if (bins) {
    ncol = bins;
  } else {
    ncol = (nnn * (nnn - 1)) / 2;
  }
  ncol += nqlist;
  size_peratom_cols = ncol;

  nmax = 0;
  maxneigh = 0;
}

/* ---------------------------------------------------------------------- */

ComputeAngleAtom::~ComputeAngleAtom()
{
  if (copymode) return;

  memory->destroy(distsq);
  memory->destroy(rlist);
  memory->destroy(nearest);
}

/* ---------------------------------------------------------------------- */

void ComputeAngleAtom::init()
{
  if (force->pair == nullptr)
    error->all(FLERR,"Compute angle/atom requires a "
               "pair style be defined");
  if (cutsq == 0.0) cutsq = force->pair->cutforce * force->pair->cutforce;
  else if (sqrt(cutsq) > force->pair->cutforce)
    error->all(FLERR,"Compute angle/atom cutoff is "
               "longer than pairwise cutoff");

  // need an occasional full neighbor list

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 1;

  // int count = 0;
  // for (int i = 0; i < modify->ncompute; i++)
  //   if (strcmp(modify->compute[i]->style,"angle/atom") == 0) count++;
  // if (count > 1 && comm->me == 0)
  //   error->warning(FLERR,"More than one compute angle/atom");
}

/* ---------------------------------------------------------------------- */

void ComputeAngleAtom::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeAngleAtom::compute_peratom()
{
  // error->warning(FLERR,fmt::format("Fine"));
  int i,j,ii,jj,inum,jnum;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int *ilist,*jlist,*numneigh,**firstneigh;

  invoked_peratom = update->ntimestep;

  // grow order parameter array if necessary

  if (atom->nmax > nmax) {
    memory->destroy(angle_array);
    nmax = atom->nmax;
    memory->create(angle_array,nmax,ncol,"angle/atom:qnarray");
    array_atom = angle_array;
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
  memset(&angle_array[0][0],0,nmax*ncol*sizeof(double));

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    double* angle = angle_array[i];
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
        memory->destroy(nearest);
        maxneigh = jnum;
        memory->create(distsq,maxneigh,"angle/atom:distsq");
        memory->create(rlist,maxneigh,3,"angle/atom:rlist");
        memory->create(nearest,maxneigh,"angle/atom:nearest");
        if (nnn == SANN) {
          memory->destroy(sort);
          sort = new Sort[jnum];
        }
      }

      if (nnn == VORO) {
        if (!(c_voronoi->invoked_flag & INVOKED_LOCAL)) {
          c_voronoi->compute_local();
          c_voronoi->invoked_flag |= INVOKED_LOCAL;
        }
        voro_local = c_voronoi->array_local;
        // error->warning(FLERR,fmt::format("Voro rows: {}", c_voronoi->size_local_rows));
      }
      

      // loop over list of all neighbors within force cutoff
      // distsq[] = distance sq to each
      // rlist[] = distance vector to each
      // alist[] = relative face area of each
      // nearest[] = atom indices of neighbors

      int ncount = 0;
      if (nnn == VORO) {
        jnum = c_voronoi->size_local_rows;
      }
      double surface = 0.0;

      for (jj = 0; jj < jnum; jj++) {

        if (nnn == VORO) {
          int i_ = (int) voro_local[jj][0];
          if (i == i_) {
            j = (int) voro_local[jj][1];
            surface += voro_local[jj][2];
            if (j >= 0) {
              alist[ncount] = voro_local[jj][2];
            } else {
              continue;
            }
          } else {
            continue;
          }
        } else {
          j = jlist[jj];
        }

        j &= NEIGHMASK;

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;
        if (rsq < cutsq) {
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
          angle[jj] = 0.0;
        continue;
      }

      // if nnn > 0, use only nearest nnn neighbors

      double rcut;
      if (nnn > 0) {
        select3(nnn,ncount,distsq,nearest,rlist);
        ncount = nnn;
      } else if (nnn == SANN) {
        // build sort structure
        for (int j = 0; j < ncount; j++) {
          sort[j].distsq = distsq[j];
          sort[j].nearest = nearest[j];
          sort[j].rlist[0] = rlist[j][0];
          sort[j].rlist[1] = rlist[j][1];
          sort[j].rlist[2] = rlist[j][2];
        }

        // sort all neighbors by distance
        qsort(sort,ncount,sizeof(Sort),compare);

        // read sort structure
        for (int j = 0; j < ncount; j++) {
          distsq[j] = sort[j].distsq;
          nearest[j] = sort[j].nearest;
          rlist[j][0] = sort[j].rlist[0];
          rlist[j][1] = sort[j].rlist[1];
          rlist[j][2] = sort[j].rlist[2];
        }

        // select solid angle based nearest neighbors
        int k = 3;
        double rsum = sqrt(distsq[0]) + sqrt(distsq[1]) + sqrt(distsq[2]);
        double r;
        for (int j = 3; j < ncount; j++) {
          r = sqrt(distsq[j]);
          rcut = rsum / (k - 2);
          if (rcut > r) {
            k++;
            rsum += r;
          } else {
            break;
          }
        }
        ncount = k;
      }

      calc_angle(rlist, ncount, angle);
    }
  }
}

/* ----------------------------------------------------------------------
   compare two neighbors I and J in sort data structure
   called via qsort in post_force() method
   is a static method so can't access sort data structure directly
   return -1 if I < J, 0 if I = J, 1 if I > J
   do comparison based on rsq distance
------------------------------------------------------------------------- */

int ComputeAngleAtom::compare(const void *pi, const void *pj)
{
  ComputeAngleAtom::Sort *ineigh = (ComputeAngleAtom::Sort *) pi;
  ComputeAngleAtom::Sort *jneigh = (ComputeAngleAtom::Sort *) pj;

  if (ineigh->distsq < jneigh->distsq) return -1;
  else if (ineigh->distsq > jneigh->distsq) return 1;
  return 0;
}


int ComputeAngleAtom::compare_angle(const void *pi, const void *pj)
{
  double anglei = *(double*)pi;
  double anglej = *(double*)pj;

  if (anglei < anglej) return -1;
  else if (anglei > anglej) return 1;
  return 0;
}


/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeAngleAtom::memory_usage()
{
  double bytes = ncol*nmax * sizeof(double);
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
  } while(0)

#define ISWAP(a,b) do {        \
    itmp = a; a = b; b = itmp; \
  } while(0)

#define SWAP3(a,b) do {                  \
    tmp = a[0]; a[0] = b[0]; b[0] = tmp; \
    tmp = a[1]; a[1] = b[1]; b[1] = tmp; \
    tmp = a[2]; a[2] = b[2]; b[2] = tmp; \
  } while(0)

/* ---------------------------------------------------------------------- */

void ComputeAngleAtom::select3(int k, int n, double *arr, int *iarr, double **arr3)
{
  int i,ir,j,l,mid,ia,itmp;
  double a,tmp,a3[3];

  // if (k > n) { // select all
  //   k = n;
  // }

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
   calculate the bond angles
------------------------------------------------------------------------- */

void ComputeAngleAtom::calc_angle(double **rlist, int ncount, double angles[])
{
  int m = 0;
  int anglecount = 0;
  for (m = 0; m < ncol; m++) {
    angles[m] == 0.0;
  }
  
  for(int ineigh = 0; ineigh < ncount-1; ineigh++) {
    for(int jneigh = ineigh + 1; jneigh < ncount; jneigh++) {
      const double * const ri = rlist[ineigh];
      const double * const rj = rlist[jneigh];
      double rmagi = dist(ri);
      double rmagj = dist(rj);
      if(rmagi <= MY_EPSILON || rmagj <= MY_EPSILON) {
        return;
      }

      // compute angle
      double c;
      c = ri[0]*rj[0] + ri[1]*rj[1] + ri[2]*rj[2];
      c /= rmagi*rmagj;
      if (c > 1.0) c = 1.0;
      if (c < -1.0) c = -1.0;
      double theta = acos(c);

      // compute histogram
      if (bins) {
        m = floor(0.5 * (c + 1.0) * bins);
        angles[m]++;
        anglecount++;
      } else {
        angles[m++] = c;
      }

      // compute Fourier components
      for (int il = 0; il < nqlist; il++) {
        int l = qlist[il];
        angles[m++] += cos(l * theta);
      }
  
    }
  }

  if (bins) {
    for (m = 0; m < ncol; m++) {
      angles[m] /= anglecount;
    }
  } else {
    qsort(angles, (nnn * (nnn - 1)) / 2, sizeof(double), compare_angle);
  }
  
}

/* ----------------------------------------------------------------------
   calculate scalar distance
------------------------------------------------------------------------- */

double ComputeAngleAtom::dist(const double r[])
{
  return sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
}
