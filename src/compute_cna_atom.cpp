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
   Contributing author: Wan Liang (Chinese Academy of Sciences)
                        Aidan Thompson (SNL)
                        Axel Kohlmeyer (Temple U)
                        Koenraad Janssens and David Olmsted (SNL)
                        Willem Gispen (UU)
------------------------------------------------------------------------- */

#include "compute_cna_atom.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
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

#define MAXNEAR 16
#define MAXCOMMON 8
#define ADAPTFCC -6
#define ADAPTBCC -7
#define NNNADAPTFCC 12
#define NNNADAPTBCC1 8
#define NNNADAPTBCC2 6
#define SANN -8
#define VORO -9
#define SIGALL 2
#define CUTSQEPSILON 1.0e-6

enum{UNKNOWN,FCC,HCP,BCC,ICOS,OTHER};
enum{NCOMMON,NBOND,MAXBOND,MINBOND};

/* ---------------------------------------------------------------------- */

ComputeCNAAtom::ComputeCNAAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  list(nullptr), nearest(nullptr), nnearest(nullptr), pattern(nullptr),
  distsq(nullptr), cutsq_(nullptr), cna_array(nullptr)
{
  if (narg < 4) error->all(FLERR,"Illegal compute cna/atom command");

  peratom_flag = 1;
  size_peratom_cols = 1;
  patternflag = 1;
  sigflag = 0;

  double cutoff = utils::numeric(FLERR,arg[3],false,lmp);
  if (cutoff < 0.0) error->all(FLERR,"Illegal compute cna/atom command");
  cutsq = cutoff*cutoff;

  int iarg = 4;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"nnn") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute cna/atom command");
      if (strcmp(arg[iarg+1],"NULL") == 0) {
        nnn = 0;
      } else if (strcmp(arg[iarg+1],"ADAPTFCC") == 0) {
        nnn = ADAPTFCC;
      } else if (strcmp(arg[iarg+1],"ADAPTBCC") == 0) {
        nnn = ADAPTBCC;
      } else if (strcmp(arg[iarg+1],"SANN") == 0) {
        nnn = SANN;
      } else if (strcmp(arg[iarg+1],"VORO") == 0) {
        nnn = VORO;
        error->all(FLERR,"Illegal compute cna/atom command");
      } else {
        nnn = utils::numeric(FLERR,arg[iarg+1],false,lmp);
        if (nnn <= 0)
          error->all(FLERR,"Illegal compute cna/atom command");
      }
      iarg += 2;
    } else if (strcmp(arg[iarg],"pattern") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute cna/atom command");
      if (strcmp(arg[iarg+1],"yes") == 0) patternflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) patternflag = 0;
      else error->all(FLERR,"Illegal compute cna/atom command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"sig") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute cna/atom command");
      if (strcmp(arg[iarg+1],"yes") == 0) sigflag = 1;
      else if (strcmp(arg[iarg+1],"all") == 0) sigflag = SIGALL;
      else if (strcmp(arg[iarg+1],"no") == 0) sigflag = 0;
      else error->all(FLERR,"Illegal compute cna/atom command");
      iarg += 2;
    } else{
      error->all(FLERR,"Illegal compute cna/atom command");
    }
  }

  nmax = 0;
  size_peratom_cols = patternflag;
  if (sigflag){
    if (sigflag == SIGALL) {
      size_peratom_cols += MAXNEAR * 4;
    } else {
      size_peratom_cols += MAXNEAR;
    } 
  }
   
}

/* ---------------------------------------------------------------------- */

ComputeCNAAtom::~ComputeCNAAtom()
{
  if (copymode) return;
 
  memory->destroy(nearest);
  memory->destroy(nnearest);
  memory->destroy(pattern);
  memory->destroy(distsq);
  memory->destroy(cutsq_);
  memory->destroy(cna_array);
}

/* ---------------------------------------------------------------------- */

void ComputeCNAAtom::init()
{
  if (force->pair == nullptr)
    error->all(FLERR,"Compute cna/atom requires a pair style be defined");
  if (sqrt(cutsq) > force->pair->cutforce)
    error->all(FLERR,"Compute cna/atom cutoff is longer than pairwise cutoff");

  // cannot use neighbor->cutneighmax b/c neighbor has not yet been init

  if (2.0*sqrt(cutsq) > force->pair->cutforce + neighbor->skin &&
      comm->me == 0)
    error->warning(FLERR,"Compute cna/atom cutoff may be too large to find "
                   "ghost atom neighbors");

  int count = 0;
  for (int i = 0; i < modify->ncompute; i++)
    if (strcmp(modify->compute[i]->style,"cna/atom") == 0) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR,"More than one compute cna/atom defined");

  // need an occasional full neighbor list

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 1;
}

/* ---------------------------------------------------------------------- */

void ComputeCNAAtom::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeCNAAtom::compute_peratom()
{
  int i,j,k,ii,jj,kk,m,n,inum,jnum,inear,jnear;
  int firstflag,ncommon,nbonds,maxbonds,minbonds;
  int nfcc,nhcp,nbcc4,nbcc6,nico,cj,ck,cl,cm;
  int *ilist,*jlist,*numneigh,**firstneigh;
  int cna[MAXNEAR][4],onenearest[10*MAXNEAR];
  double onedistsq[10*MAXNEAR];
  int common[MAXCOMMON],bonds[MAXCOMMON];
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;

  invoked_peratom = update->ntimestep;

  // grow arrays if necessary

  if (atom->nmax > nmax) {
    memory->destroy(nearest);
    memory->destroy(distsq);
    memory->destroy(cutsq_);
    memory->destroy(cna_array);
    memory->destroy(nnearest);
    memory->destroy(pattern);
    nmax = atom->nmax;

    memory->create(nearest,nmax,10*MAXNEAR,"cna:nearest");
    memory->create(distsq,nmax,10*MAXNEAR,"cna:distsq");
    memory->create(cutsq_,nmax,"cna:cutsq_");
    memory->create(cna_array,nmax,size_peratom_cols,"cna/atom:cna_array");
    array_atom = cna_array;
    memory->create(nnearest,nmax,"cna:nnearest");
    memory->create(pattern,nmax,"cna:cna_pattern");
    // vector_atom = pattern;
  }

  // invoke full neighbor list (will copy or build if necessary)

  neighbor->build_one(list);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // find the neighbors of each atom within cutoff using full neighbor list
  // nearest[] = atom indices of nearest neighbors, up to MAXNEAR
  // do this for all atoms, not just compute group
  // since CNA calculation requires neighbors of neighbors

  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  memset(&cna_array[0][0],0,nmax*size_peratom_cols*sizeof(double));

  int nerror = 0;
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    find_neighbors(i, x, jlist, jnum, cutsq, nearest[i], distsq[i], &n);
    nnearest[i] = n;
    // if (n > 0) {
    //   error->warning(FLERR,fmt::format("NN =  {}",n),0);
    // }

    if (nnn != 0) {
      compute_acutsq(distsq[i], n, &cutsq_[i], &nnearest[i]);
      select3(nnearest[i], n, distsq[i], nearest[i]);
      // if (nnearest[i] > 0) {
      //   error->warning(FLERR,fmt::format("NN =  {}",nnearest[i]),0);
      // }
    }
  }
  

  // warning message

  int nerrorall;
  MPI_Allreduce(&nerror,&nerrorall,1,MPI_INT,MPI_SUM,world);
  if (nerrorall && comm->me == 0)
    error->warning(FLERR,fmt::format("Too many neighbors in CNA for {} "
                                     "atoms",nerrorall),0);

  // compute CNA for each atom in group
  // if nnn==0, then only performed if # of nearest neighbors = 12 or 14 (fcc,hcp)

  nerror = 0;
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];

    if (!(mask[i] & groupbit)) {
      pattern[i] = UNKNOWN;
      continue;
    }

    // if (nnn == 0) {
    //   if (nnearest[i] != 12 && nnearest[i] != 14) {
    //     pattern[i] = OTHER;
    //     continue;
    //   }
    // }

    // loop over near neighbors of I to build cna data structure
    // cna[k][NCOMMON] = # of common neighbors of I with each of its neighs
    // cna[k][NBONDS] = # of bonds between those common neighbors
    // cna[k][MAXBOND] = max # of bonds of any common neighbor
    // cna[k][MINBOND] = min # of bonds of any common neighbor

    // reset cna to zero
    for (m = 0; m < MAXNEAR; m++) {
      cna[m][NCOMMON] = cna[m][NBOND] = cna[m][MAXBOND] = cna[m][MINBOND] = 0;
    }

    for (m = 0; m < nnearest[i]; m++) {
      j = nearest[i][m];

      // common = list of neighbors common to atom I and atom J
      // if J is an owned atom, use its near neighbor list to find them
      // if J is a ghost atom, use full neighbor list of I to find them
      // in latter case, must exclude J from I's neighbor list

      // TODO: ensure bidirectional edges are accounted for
      if (j < nlocal) {
        firstflag = 1;
        ncommon = 0;
        for (inear = 0; inear < nnearest[i]; inear++)
          for (jnear = 0; jnear < nnearest[j]; jnear++)
            if (nearest[i][inear] == nearest[j][jnear]) {
              if (ncommon < MAXCOMMON) common[ncommon++] = nearest[i][inear];
              else if (firstflag) {
                nerror++;
                firstflag = 0;
              }
            }

      } else {
        jlist = firstneigh[i];
        jnum = numneigh[i];
        find_neighbors(j, x, jlist, jnum, cutsq, onenearest, onedistsq, &n);

        if (nnn != 0) {
          int n_;
          double dummy;
          compute_acutsq(onedistsq, n, &dummy, &n_);
          select3(n_, n, onedistsq, onenearest);
          n = n_;
          // if (n > 0) {
          //   error->warning(FLERR,fmt::format("NN =  {}",n),0);
          // }
        }

        firstflag = 1;
        ncommon = 0;
        for (inear = 0; inear < nnearest[i]; inear++)
          for (jnear = 0; (jnear < n) && (n < MAXNEAR); jnear++)
            if (nearest[i][inear] == onenearest[jnear]) {
              if (ncommon < MAXCOMMON) common[ncommon++] = nearest[i][inear];
              else if (firstflag) {
                nerror++;
                firstflag = 0;
              }
            }
      }

      cna[m][NCOMMON] = ncommon;

      // calculate total # of bonds between common neighbor atoms
      // also max and min # of common atoms any common atom is bonded to
      // bond = pair of atoms within cutoff

      for (n = 0; n < ncommon; n++) bonds[n] = 0;

      nbonds = 0;
      for (jj = 0; jj < ncommon; jj++) {
        j = common[jj];
        xtmp = x[j][0];
        ytmp = x[j][1];
        ztmp = x[j][2];

        double acutsq = cutsq;
        if (nnn != 0) {
          acutsq = cutsq_[j];
          if (acutsq < 2 * CUTSQEPSILON) {
            jlist = firstneigh[i];
            jnum = numneigh[i];
            int nn;
            find_neighbors(j, x, jlist, jnum, cutsq, onenearest, onedistsq, &n);
            compute_acutsq(onedistsq, n, &acutsq, &nn);
            acutsq = 1.4 * 1.4;
          }
        }

        for (kk = 0; kk < ncommon; kk++) {
          if (jj == kk || nnn == 0 && jj > kk) continue;
          k = common[kk];
          delx = xtmp - x[k][0];
          dely = ytmp - x[k][1];
          delz = ztmp - x[k][2];
          rsq = delx*delx + dely*dely + delz*delz;
          if (rsq < acutsq) {
            nbonds++;
            bonds[jj]++;
            bonds[kk]++;
          }
        }
      }

      cna[m][NBOND] = nbonds;

      maxbonds = 0;
      minbonds = MAXCOMMON;
      for (n = 0; n < ncommon; n++) {
        maxbonds = MAX(bonds[n],maxbonds);
        minbonds = MIN(bonds[n],minbonds);
      }
      cna[m][MAXBOND] = maxbonds;
      cna[m][MINBOND] = minbonds;
    }

    // detect CNA pattern of the atom

    nfcc = nhcp = nbcc4 = nbcc6 = nico = 0;
    pattern[i] = OTHER;

    if (nnearest[i] == 12) {
      for (inear = 0; inear < 12; inear++) {
        cj = cna[inear][NCOMMON];
        ck = cna[inear][NBOND];
        cl = cna[inear][MAXBOND];
        cm = cna[inear][MINBOND];
        if (cj == 4 && ck == 2 && cl == 1 && cm == 1) nfcc++;
        else if (cj == 4 && ck == 2 && cl == 2 && cm == 0) nhcp++;
        else if (cj == 5 && ck == 5 && cl == 2 && cm == 2) nico++;
      }
      if (nfcc == 12) pattern[i] = FCC;
      else if (nfcc == 6 && nhcp == 6) pattern[i] = HCP;
      else if (nico == 12) pattern[i] = ICOS;

    } else if (nnearest[i] == 14) {
      for (inear = 0; inear < 14; inear++) {
        cj = cna[inear][NCOMMON];
        ck = cna[inear][NBOND];
        cl = cna[inear][MAXBOND];
        cm = cna[inear][MINBOND];
        if (cj == 4 && ck == 4 && cl == 2 && cm == 2) nbcc4++;
        else if (cj == 6 && ck == 6 && cl == 2 && cm == 2) nbcc6++;
      }
      if (nbcc4 == 6 && nbcc6 == 8) pattern[i] = BCC;
    }

    // output cna as array/atom
    double* cna_ = cna_array[i];
    int mm = 0;
    if (patternflag) {
      cna_[mm++] = pattern[i];
    }
    if (sigflag) {
      // row-sort ascending
      qsort(cna,MAXNEAR,4*sizeof(int),compare_cna);

      for (m = 0; m < MAXNEAR; m++) {
        cna_[mm++] = cna[m][NCOMMON];
        if (sigflag == SIGALL) {
          cna_[mm++] = cna[m][NBOND];
          cna_[mm++] = cna[m][MAXBOND];
          cna_[mm++] = cna[m][MINBOND];
        }
      }
    }
  }

  // warning message

  MPI_Allreduce(&nerror,&nerrorall,1,MPI_INT,MPI_SUM,world);
  if (nerrorall && comm->me == 0)
    error->warning(FLERR,fmt::format("Too many common neighbors in CNA {} "
                                     "times", nerrorall));
}



/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeCNAAtom::memory_usage()
{
  double bytes = nmax * sizeof(int);
  bytes += nmax * MAXNEAR * sizeof(int);
  bytes += nmax * sizeof(double);

  if (nnn != 0) {
    bytes += nmax * MAXNEAR * 4 * sizeof(double);
  }
  
  return bytes;
}


/* ----------------------------------------------------------------------
   compare two signatures in cna array
   called via qsort in compute_peratom()
   return -1 if I < J, 0 if I = J, 1 if I > J
   do comparison based on dictionary order
------------------------------------------------------------------------- */

int ComputeCNAAtom::compare_cna(const void *pi, const void *pj)
{
  // read arrays
  int *arri = (int*)pi;
  int *arrj = (int*)pj;

  // order of comparison
  int comparr[4] = {NCOMMON, NBOND, MAXBOND, MINBOND};

  for (int i = 0; i < 4; i++) {
    int comp = comparr[i];
    if (arri[comp] < arrj[comp]) return -1;
    else if (arri[comp] > arrj[comp]) return 1;
  }
  return 0;
}


/* ----------------------------------------------------------------------
   compare two neighbors
   called via qsort in compute_peratom()
   return -1 if I < J, 0 if I = J, 1 if I > J
   do comparison based on squared distance
------------------------------------------------------------------------- */

int ComputeCNAAtom::compare_neigh(const void *pi, const void *pj)
{
  double distsqi = *(double*)pi;
  double distsqj = *(double*)pj;

  if (distsqi < distsqj) return -1;
  else if (distsqi > distsqj) return 1;
  return 0;
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

/* ---------------------------------------------------------------------- */

void ComputeCNAAtom::select3(int k, int n, double *arr, int *iarr)
{
  int i,ir,j,l,mid,ia,itmp;
  double a,tmp;

  if (k > n) { // select all
    k = n;
  }

  arr--;
  iarr--;
  l = 1;
  ir = n;
  for (;;) {
    if (ir <= l+1) {
      if (ir == l+1 && arr[ir] < arr[l]) {
        SWAP(arr[l],arr[ir]);
        ISWAP(iarr[l],iarr[ir]);
      }
      return;
    } else {
      mid=(l+ir) >> 1;
      SWAP(arr[mid],arr[l+1]);
      ISWAP(iarr[mid],iarr[l+1]);
      if (arr[l] > arr[ir]) {
        SWAP(arr[l],arr[ir]);
        ISWAP(iarr[l],iarr[ir]);
      }
      if (arr[l+1] > arr[ir]) {
        SWAP(arr[l+1],arr[ir]);
        ISWAP(iarr[l+1],iarr[ir]);
      }
      if (arr[l] > arr[l+1]) {
        SWAP(arr[l],arr[l+1]);
        ISWAP(iarr[l],iarr[l+1]);
      }
      i = l+1;
      j = ir;
      a = arr[l+1];
      ia = iarr[l+1];
      for (;;) {
        do i++; while (arr[i] < a);
        do j--; while (arr[j] > a);
        if (j < i) break;
        SWAP(arr[i],arr[j]);
        ISWAP(iarr[i],iarr[j]);
      }
      arr[l+1] = arr[j];
      arr[j] = a;
      iarr[l+1] = iarr[j];
      iarr[j] = ia;
      if (j >= k) ir = j-1;
      if (j <= k) l = i;
    }
  }
}

// Find neighbors of atom id within cutoff (cutsq)
// uses jlist (of length jnum) for neighbor ids, x for positions
// stores neighbor ids in nearest, distances in distsq, and number in *pn

void ComputeCNAAtom::find_neighbors(int id, double **x, int *jlist, int jnum, double cutsq,
                                    int *onenearest, double *onedistsq, int *pn)
{
  double xtmp, ytmp, ztmp, delx, dely, delz, rsq;
  int n, k, kk;
  xtmp = x[id][0];
  ytmp = x[id][1];
  ztmp = x[id][2];

  n = 0;
  for (kk = 0; kk < jnum; kk++) {
    k = jlist[kk];
    k &= NEIGHMASK;
    if (k == id) continue;

    delx = xtmp - x[k][0];
    dely = ytmp - x[k][1];
    delz = ztmp - x[k][2];
    rsq = delx*delx + dely*dely + delz*delz;

    if (rsq < cutsq) {          
      if (nnn == 0) {
        if (n < MAXNEAR) onenearest[n++] = k;
        else break;
      } else {
        onedistsq[n] = rsq;
        onenearest[n++] = k;
      }
    }
  }
  *pn = n;
}

// Compute adaptive cutoff using one of three different algorithms.
// Stores adaptive cutoff in *p_acutsq
// Stores number of neighbors within adaptive cutoff in *p_nnn

void ComputeCNAAtom::compute_acutsq(double *distsq_, int m, double *p_acutsq, int *p_nnn)
{
  // copy so distsq_ is not changed
  double distsq__[m];
  for (int i = 0; i < m; i++){
    distsq__[i] = distsq_[i];
  }

  qsort(distsq__, m, sizeof(double), compare_neigh);
  double acutsq = 0.0;
  int n;

  if (nnn == ADAPTFCC) {
    // compute acutsq
    for (n = 0; n < NNNADAPTFCC; n++) {
      acutsq += distsq__[n];
    }
    acutsq *= 0.5 * (1.0 + sqrt(2)) / NNNADAPTFCC;
    // compute nnn
    for (n = NNNADAPTFCC; n < MAXNEAR; n++) {
      double rsq = distsq__[n];
      if (rsq > acutsq) {
        break;
      }
    }
  } else if (nnn == ADAPTBCC) {
    // compute acutsq
    for (n = 0; n < NNNADAPTBCC1; n++) {
      acutsq += 2 / sqrt(3) * distsq__[n] / NNNADAPTBCC1;
      // acutsq += 1 / sqrt(3) * distsq__[n] / NNNADAPTBCC1;
    }
    for (n = NNNADAPTBCC1; n < NNNADAPTBCC1 + NNNADAPTBCC2; n++) {
      acutsq += distsq__[n] / NNNADAPTBCC2;
    }
    acutsq *= 0.25 * (1.0 + sqrt(2));
    // compute nnn
    for (n = NNNADAPTBCC1 + NNNADAPTBCC2; n < MAXNEAR; n++) {
      double rsq = distsq__[n];
      if (rsq > acutsq) {
        break;
      }
    }
  } else if (nnn == SANN) {
    double rsum = sqrt(distsq__[0]) + sqrt(distsq__[1]) + sqrt(distsq__[2]);
    double r, rcut;
    for (n = 3; n < MAXNEAR; n++) {
      r = sqrt(distsq__[n]);
      rcut = rsum / (n - 2);
      if (r > rcut) {
        break;
      } else {
        rsum += r;
      }
    }
    acutsq = rcut*rcut;
  } else {
    for (n = 0; n < nnn; n++) {
      acutsq = MAX(distsq__[n], acutsq);
    }
    acutsq += CUTSQEPSILON;
  }

  *p_nnn = n;
  *p_acutsq = acutsq;
}