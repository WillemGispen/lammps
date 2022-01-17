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

#include "compute_coord_atom.h"

#include "atom.h"
#include "comm.h"
#include "compute_orientorder_atom.h"
#include "error.h"
#include "force.h"
#include "group.h"
// #include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"

#include <cmath>
#include <cstring>


using namespace LAMMPS_NS;

#define INVOKED_PERATOM 8
#define NNN 12
#define NNNMAX 512
#define ALLCOMP -21
#define MY_4PI 12.56637061435917246
#define QEPSILON 1.0e-6

/* ---------------------------------------------------------------------- */

ComputeCoordAtom::ComputeCoordAtom(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  typelo(nullptr), typehi(nullptr), cvec(nullptr), carray(nullptr),
  group2(nullptr), id_orientorder(nullptr), normv(nullptr),
  qnm_r(nullptr), qnm_i(nullptr), cglist(nullptr)
{
  if (narg < 5) error->all(FLERR,"Illegal compute coord/atom command");

  jgroup = group->find("all");
  jgroupbit = group->bitmask[jgroup];
  cstyle = NONE;

  if (strcmp(arg[3],"cutoff") == 0) {
    cstyle = CUTOFF;
    double cutoff = utils::numeric(FLERR,arg[4],false,lmp);
    cutsq = cutoff*cutoff;

    int iarg = 5;
    if ((narg > 6) && (strcmp(arg[5],"group") == 0)) {
      int len = strlen(arg[6])+1;
      group2 = new char[len];
      strcpy(group2,arg[6]);
      iarg += 2;
      jgroup = group->find(group2);
      if (jgroup == -1)
        error->all(FLERR,"Compute coord/atom group2 ID does not exist");
      jgroupbit = group->bitmask[jgroup];
    }

    ncol = narg-iarg + 1;
    int ntypes = atom->ntypes;
    typelo = new int[ncol];
    typehi = new int[ncol];

    if (narg == iarg) {
      ncol = 1;
      typelo[0] = 1;
      typehi[0] = ntypes;
    } else {
      ncol = 0;
      while (iarg < narg) {
        utils::bounds(FLERR,arg[iarg],1,ntypes,typelo[ncol],typehi[ncol],error);
        if (typelo[ncol] > typehi[ncol])
          error->all(FLERR,"Illegal compute coord/atom command");
        ncol++;
        iarg++;
      }
    }

  } else if (strcmp(arg[3],"orientorder") == 0) {
    cstyle = ORIENT;
    if (narg != 6) error->all(FLERR,"Illegal compute coord/atom command");

    int n = strlen(arg[4]) + 1;
    id_orientorder = new char[n];
    strcpy(id_orientorder,arg[4]);

    int iorientorder = modify->find_compute(id_orientorder);
    if (iorientorder < 0)
      error->all(FLERR,"Could not find compute coord/atom compute ID");
    if (!utils::strmatch(modify->compute[iorientorder]->style,"^orientorder/atom"))
      error->all(FLERR,"Compute coord/atom compute ID is not orientorder/atom");
    c_orientorder = (ComputeOrientOrderAtom*)(modify->compute[iorientorder]);

    if (strcmp(arg[5],"output") == 0) {
      nqlist = c_orientorder->nqlist;
      ncol = NNN * (1 + 2 * nqlist);
    } else{
      threshold = utils::numeric(FLERR,arg[5],false,lmp);
      if (threshold <= -1.0 || threshold >= 1.0)
        error->all(FLERR,"Compute coord/atom threshold not between -1 and 1");

      ncol = 1;
      typelo = new int[ncol];
      typehi = new int[ncol];
      typelo[0] = 1;
      typehi[0] = atom->ntypes;
    }

  } else error->all(FLERR,"Invalid cstyle in compute coord/atom");

  peratom_flag = 1;
  if (ncol == 1) size_peratom_cols = 0;
  else size_peratom_cols = ncol;

  nmax = 0;
}

/* ---------------------------------------------------------------------- */

ComputeCoordAtom::~ComputeCoordAtom()
{
  if (copymode) return;

  delete [] group2;
  delete [] typelo;
  delete [] typehi;
  memory->destroy(cvec);
  memory->destroy(carray);
  delete [] id_orientorder;
  memory->destroy(qnm_r);
  memory->destroy(qnm_i);
  memory->destroy(cglist);
}

/* ---------------------------------------------------------------------- */

void ComputeCoordAtom::init()
{
  if (cstyle == ORIENT) {
    int iorientorder = modify->find_compute(id_orientorder);
    c_orientorder = (ComputeOrientOrderAtom*)(modify->compute[iorientorder]);
    cutsq = c_orientorder->cutsq;
    l = c_orientorder->qlcomp;
    //  communicate real and imaginary 2*l+1 components of the normalized vector
    comm_forward = 2*(2*l+1);
    if (!(c_orientorder->qlcompflag))
      error->all(FLERR,"Compute coord/atom requires components "
                 "option in compute orientorder/atom");
  }

  if (force->pair == nullptr)
    error->all(FLERR,"Compute coord/atom requires a pair style be defined");
  if (sqrt(cutsq) > force->pair->cutforce)
    error->all(FLERR,
               "Compute coord/atom cutoff is longer than pairwise cutoff");

  int qmax = 12;
  memory->create(qnm_r,nqlist,2*qmax+1,"coord/atom:qnm_r");
  memory->create(qnm_i,nqlist,2*qmax+1,"coord/atom:qnm_i");

  // need an occasional full neighbor list

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->compute = 1;
  neighbor->requests[irequest]->half = 0;
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 1;

  int wlflag = 1;
  if (wlflag) init_clebsch_gordan();
}

/* ---------------------------------------------------------------------- */

void ComputeCoordAtom::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void ComputeCoordAtom::compute_peratom()
{
  int i,j,m,ii,jj,kk,inum,jnum,jtype,n;
  double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double *count;

  invoked_peratom = update->ntimestep;

  // grow coordination array if necessary

  if (atom->nmax > nmax) {
    if (ncol == 1) {
      memory->destroy(cvec);
      nmax = atom->nmax;
      memory->create(cvec,nmax,"coord/atom:cvec");
      vector_atom = cvec;
    } else {
      memory->destroy(carray);
      nmax = atom->nmax;
      memory->create(carray,nmax,ncol,"coord/atom:carray");
      array_atom = carray;
    }
  }

  if (cstyle == ORIENT) {
    if (!(c_orientorder->invoked_flag & INVOKED_PERATOM)) {
      c_orientorder->compute_peratom();
      c_orientorder->invoked_flag |= INVOKED_PERATOM;
    }
    nqlist = c_orientorder->nqlist;
    qlist = c_orientorder->qlist;
    normv = c_orientorder->array_atom;
    comm->forward_comm_compute(this);
  }

  // invoke full neighbor list (will copy or build if necessary)

  neighbor->build_one(list);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // compute coordination number(s) for each atom in group
  // use full neighbor list to count atoms less than cutoff

  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;

  if (cstyle == CUTOFF) {

    if (ncol == 1) {

      for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        if (mask[i] & groupbit) {
          xtmp = x[i][0];
          ytmp = x[i][1];
          ztmp = x[i][2];
          jlist = firstneigh[i];
          jnum = numneigh[i];

          n = 0;
          for (jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            j &= NEIGHMASK;

            if (mask[j] & jgroupbit) {
              jtype = type[j];
              delx = xtmp - x[j][0];
              dely = ytmp - x[j][1];
              delz = ztmp - x[j][2];
              rsq = delx*delx + dely*dely + delz*delz;
              if (rsq < cutsq && jtype >= typelo[0] && jtype <= typehi[0])
                n++;
            }
          }

          cvec[i] = n;
        } else cvec[i] = 0.0;
      }

    } else {
      for (ii = 0; ii < inum; ii++) {
        i = ilist[ii];
        count = carray[i];
        for (m = 0; m < ncol; m++) count[m] = 0.0;

        if (mask[i] & groupbit) {
          xtmp = x[i][0];
          ytmp = x[i][1];
          ztmp = x[i][2];
          jlist = firstneigh[i];
          jnum = numneigh[i];


          for (jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            j &= NEIGHMASK;

            jtype = type[j];
            delx = xtmp - x[j][0];
            dely = ytmp - x[j][1];
            delz = ztmp - x[j][2];
            rsq = delx*delx + dely*dely + delz*delz;
            if (rsq < cutsq) {
              for (m = 0; m < ncol; m++)
                if (jtype >= typelo[m] && jtype <= typehi[m])
                  count[m] += 1.0;
            }
          }
        }
      }
    }

  } else if (cstyle == ORIENT) {

    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      count = carray[i];
      for (int il = 0; il < nqlist; il++) count[il] = 0.0;

      if (mask[i] & groupbit) {
        xtmp = x[i][0];
        ytmp = x[i][1];
        ztmp = x[i][2];
        jlist = firstneigh[i];
        jnum = numneigh[i];

        tagint jtag;
        tagint *tag = atom->tag;

        int ncount;
        double distsq[NNNMAX];
        int nearest[NNNMAX];
        for (int ncount = 0; ncount < NNNMAX; ncount++) {
          distsq[ncount] = 100.0;
          nearest[ncount] = -1;
        }

        for (jj = 0, ncount=0; jj < jnum && ncount < NNNMAX; jj++) {
          j = jlist[jj];
          j &= NEIGHMASK;
          delx = xtmp - x[j][0];
          dely = ytmp - x[j][1];
          delz = ztmp - x[j][2];
          rsq = delx*delx + dely*dely + delz*delz;
          if (rsq < 1.4 * 1.4 && j >= 0) {
            distsq[ncount] = rsq;
            nearest[ncount] = j;
            ncount++;
          }
        }

        select3(NNN, NNNMAX, distsq, nearest);

        for (jj = 0, kk = 0; jj < NNN; jj++) {
          // write tag
          j = nearest[jj];
          if (j >= 0) jtag = tag[j];
          else jtag = 0;
          count[kk++] = jtag;

          int ldflag = 1;

          int k = nqlist;
          for (int il = 0; il < nqlist; il++) {
            int l = qlist[il];
            double qnormfac = sqrt(MY_4PI/(2*l+1));

            // calculate average/product of orientorder components over pair
            for(int m = 0; m < 2*(2*l+1); m++) {
             if (j > 0) {
              double qi = normv[i][k];
              double qj = normv[j][k];
              if (ldflag) { // average
                double qnfaci = normv[i][il]/qnormfac;
                double qnfacj = normv[j][il]/qnormfac;
                qi *= qnfaci;
                qj *= qnfacj;

                if (m % 2 == 0) { // real
                  qnm_r[il][m/2] = (qi + qj) / 2.0;
                } else { // imaginary
                  qnm_i[il][m/2] = (qi + qj) / 2.0;
                }
              } else { // product
                if (m % 2 == 0) { // real
                  qnm_r[il][m/2] = qi*qj;
                } else { // imaginary
                  qnm_i[il][m/2] = qi*qj;
                }
              }
             }
              k++;
            }
          }

          // error->warning(FLERR,fmt::format("NN =  {} {} {}",i,j,k),0);

          k = nqlist;
          for (int il = 0; il < nqlist; il++) {
            int l = qlist[il];
            double dot_product = 0.0;
            for (int m=0; m < 2*l+1; m++) {
              if (j >= 0) {
                if (ldflag) {
                  dot_product += qnm_r[il][m] * qnm_r[il][m];
                  dot_product += qnm_i[il][m] * qnm_i[il][m];
                } else {
                  dot_product += qnm_r[il][m];
                  dot_product += qnm_i[il][m];
                }
              }
            }

            if (ldflag) {
              double qnormfac = sqrt(MY_4PI/(2*l+1));
              count[kk++] = qnormfac * sqrt(dot_product);
            } else {
              count[kk++] = dot_product;
            }

            if (dot_product > threshold){
              n++;
            }
          }


          // TODO: make wlflag and ldflag options
          int wlflag = 1;
          if (wlflag) {
            int idxcg_count = 0;
            // calculate W_l
            int k = nqlist;
            for (int il = 0; il < nqlist; il++) {
              int l = qlist[il];

              double wlsum = 0.0;
              for(int m1 = 0; m1 < 2*l+1; m1++) {
                for(int m2 = MAX(0,l-m1); m2 < MIN(2*l+1,3*l-m1+1); m2++) {
                  int m = m1 + m2 - l;
                  double qm1qm2_r = qnm_r[il][m1]*qnm_r[il][m2] - qnm_i[il][m1]*qnm_i[il][m2];
                  double qm1qm2_i = qnm_r[il][m1]*qnm_i[il][m2] + qnm_i[il][m1]*qnm_r[il][m2];
                  wlsum += (qm1qm2_r*qnm_r[il][m] + qm1qm2_i*qnm_i[il][m])*cglist[idxcg_count];
                  idxcg_count++;
                }
              }
              count[kk++] = wlsum/sqrt(2*l+1);
            }
          }

        }
        

        if (ncol == 1) {
          cvec[i] = n;
        }
      } else cvec[i] = 0.0;
    }
  }
}

/* ---------------------------------------------------------------------- */

int ComputeCoordAtom::pack_forward_comm(int n, int *list, double *buf,
                                        int /*pbc_flag*/, int * /*pbc*/)
{
  int i,m=0,j;
  for (i = 0; i < n; ++i) {
    for (j = nqlist; j < nqlist + 2*(2*l+1); ++j) {
      buf[m++] = normv[list[i]][j];
    }
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void ComputeCoordAtom::unpack_forward_comm(int n, int first, double *buf)
{
  int i,last,m=0,j;
  last = first + n;
  for (i = first; i < last; ++i) {
    for (j = nqlist; j < nqlist + 2*(2*l+1); ++j) {
      normv[i][j] = buf[m++];
    }
  }
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based array
------------------------------------------------------------------------- */

double ComputeCoordAtom::memory_usage()
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

/* ---------------------------------------------------------------------- */


void ComputeCoordAtom::select3(int k, int n, double *arr, int *iarr)
{
  int i,ir,j,l,mid,ia,itmp;
  double a,tmp,a3[3];

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



/* ----------------------------------------------------------------------
   assign Clebsch-Gordan coefficients
   using the quasi-binomial formula VMK 8.2.1(3)
   specialized for case j1=j2=j=l
------------------------------------------------------------------------- */

void ComputeCoordAtom::init_clebsch_gordan()
{
  double sum,dcg,sfaccg, sfac1, sfac2;
  int m, aa2, bb2, cc2;
  int ifac, idxcg_count;

  qlist = c_orientorder->qlist;

  idxcg_count = 0;
  for (int il = 0; il < nqlist; il++) {
    int l = qlist[il];
    for(int m1 = 0; m1 < 2*l+1; m1++)
      for(int m2 = MAX(0,l-m1); m2 < MIN(2*l+1,3*l-m1+1); m2++)
        idxcg_count++;
  }
  idxcg_max = idxcg_count;
  memory->create(cglist, idxcg_max, "computecoordatom:cglist");

  idxcg_count = 0;
  for (int il = 0; il < nqlist; il++) {
    int l = qlist[il];
    for(int m1 = 0; m1 < 2*l+1; m1++) {
        aa2 = m1 - l;
        for(int m2 = MAX(0,l-m1); m2 < MIN(2*l+1,3*l-m1+1); m2++) {
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

/* ----------------------------------------------------------------------
   factorial n, wrapper for precomputed table
------------------------------------------------------------------------- */

double ComputeCoordAtom::factorial(int n)
{
  if (n < 0 || n > nmaxfactorial)
    error->all(FLERR,fmt::format("Invalid argument to factorial {}", n));

  return nfac_table[n];
}

/* ----------------------------------------------------------------------
   factorial n table, size SNA::nmaxfactorial+1
------------------------------------------------------------------------- */

const double ComputeCoordAtom::nfac_table[] = {
  1,
  1,
  2,
  6,
  24,
  120,
  720,
  5040,
  40320,
  362880,
  3628800,
  39916800,
  479001600,
  6227020800,
  87178291200,
  1307674368000,
  20922789888000,
  355687428096000,
  6.402373705728e+15,
  1.21645100408832e+17,
  2.43290200817664e+18,
  5.10909421717094e+19,
  1.12400072777761e+21,
  2.5852016738885e+22,
  6.20448401733239e+23,
  1.5511210043331e+25,
  4.03291461126606e+26,
  1.08888694504184e+28,
  3.04888344611714e+29,
  8.8417619937397e+30,
  2.65252859812191e+32,
  8.22283865417792e+33,
  2.63130836933694e+35,
  8.68331761881189e+36,
  2.95232799039604e+38,
  1.03331479663861e+40,
  3.71993326789901e+41,
  1.37637530912263e+43,
  5.23022617466601e+44,
  2.03978820811974e+46,
  8.15915283247898e+47,
  3.34525266131638e+49,
  1.40500611775288e+51,
  6.04152630633738e+52,
  2.65827157478845e+54,
  1.1962222086548e+56,
  5.50262215981209e+57,
  2.58623241511168e+59,
  1.24139155925361e+61,
  6.08281864034268e+62,
  3.04140932017134e+64,
  1.55111875328738e+66,
  8.06581751709439e+67,
  4.27488328406003e+69,
  2.30843697339241e+71,
  1.26964033536583e+73,
  7.10998587804863e+74,
  4.05269195048772e+76,
  2.35056133128288e+78,
  1.3868311854569e+80,
  8.32098711274139e+81,
  5.07580213877225e+83,
  3.14699732603879e+85,
  1.98260831540444e+87,
  1.26886932185884e+89,
  8.24765059208247e+90,
  5.44344939077443e+92,
  3.64711109181887e+94,
  2.48003554243683e+96,
  1.71122452428141e+98,
  1.19785716699699e+100,
  8.50478588567862e+101,
  6.12344583768861e+103,
  4.47011546151268e+105,
  3.30788544151939e+107,
  2.48091408113954e+109,
  1.88549470166605e+111,
  1.45183092028286e+113,
  1.13242811782063e+115,
  8.94618213078297e+116,
  7.15694570462638e+118,
  5.79712602074737e+120,
  4.75364333701284e+122,
  3.94552396972066e+124,
  3.31424013456535e+126,
  2.81710411438055e+128,
  2.42270953836727e+130,
  2.10775729837953e+132,
  1.85482642257398e+134,
  1.65079551609085e+136,
  1.48571596448176e+138,
  1.3520015276784e+140,
  1.24384140546413e+142,
  1.15677250708164e+144,
  1.08736615665674e+146,
  1.03299784882391e+148,
  9.91677934870949e+149,
  9.61927596824821e+151,
  9.42689044888324e+153,
  9.33262154439441e+155,
  9.33262154439441e+157,
  9.42594775983835e+159,
  9.61446671503512e+161,
  9.90290071648618e+163,
  1.02990167451456e+166,
  1.08139675824029e+168,
  1.14628056373471e+170,
  1.22652020319614e+172,
  1.32464181945183e+174,
  1.44385958320249e+176,
  1.58824554152274e+178,
  1.76295255109024e+180,
  1.97450685722107e+182,
  2.23119274865981e+184,
  2.54355973347219e+186,
  2.92509369349301e+188,
  3.3931086844519e+190,
  3.96993716080872e+192,
  4.68452584975429e+194,
  5.5745857612076e+196,
  6.68950291344912e+198,
  8.09429852527344e+200,
  9.8750442008336e+202,
  1.21463043670253e+205,
  1.50614174151114e+207,
  1.88267717688893e+209,
  2.37217324288005e+211,
  3.01266001845766e+213,
  3.8562048236258e+215,
  4.97450422247729e+217,
  6.46685548922047e+219,
  8.47158069087882e+221,
  1.118248651196e+224,
  1.48727070609069e+226,
  1.99294274616152e+228,
  2.69047270731805e+230,
  3.65904288195255e+232,
  5.01288874827499e+234,
  6.91778647261949e+236,
  9.61572319694109e+238,
  1.34620124757175e+241,
  1.89814375907617e+243,
  2.69536413788816e+245,
  3.85437071718007e+247,
  5.5502938327393e+249,
  8.04792605747199e+251,
  1.17499720439091e+254,
  1.72724589045464e+256,
  2.55632391787286e+258,
  3.80892263763057e+260,
  5.71338395644585e+262,
  8.62720977423323e+264,
  1.31133588568345e+267,
  2.00634390509568e+269,
  3.08976961384735e+271,
  4.78914290146339e+273,
  7.47106292628289e+275,
  1.17295687942641e+278,
  1.85327186949373e+280,
  2.94670227249504e+282,
  4.71472363599206e+284,
  7.59070505394721e+286,
  1.22969421873945e+289,
  2.0044015765453e+291,
  3.28721858553429e+293,
  5.42391066613159e+295,
  9.00369170577843e+297,
  1.503616514865e+300, // nmaxfactorial = 167
};

