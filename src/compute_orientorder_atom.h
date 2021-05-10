/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS

ComputeStyle(orientorder/atom,ComputeOrientOrderAtom)

#else

#ifndef LMP_COMPUTE_ORIENTORDER_ATOM_H
#define LMP_COMPUTE_ORIENTORDER_ATOM_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeOrientOrderAtom : public Compute {
 public:
  ComputeOrientOrderAtom(class LAMMPS *, int, char **);
  ~ComputeOrientOrderAtom();
  virtual void init();
  void init_list(int, class NeighList *);
  virtual void compute_peratom();
  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);
  double memory_usage();
  double cutsq;
  int iqlcomp, qlcomp, qlcompflag, wlflag, wlhatflag;
  int nnn;
  int icompute, commflag, averageflag;
  int ncol_qn_neigh;
  const static int max_len_qn_neigh = 350; // 350, approx 2*(2l+1) for l=1, ..., 12
  int *qlist;
  int nqlist;

 protected:
  int nmax,maxneigh,ncol;
  int jjqlcomp_0;
  int *iql_;
  int *jjqlcomp_;
  class NeighList *list;
  double *distsq;
  int *nearest;
  double **rlist;
  double **qn_neigh;
  int qmax;
  double **qnarray;
  double **qnm_r;
  double **qnm_i;

  void select3(int, int, double *, int *, double **);
  void calc_boop(double **rlist, double **qn_neigh,
                 int numNeighbors, double qn[], int nlist[], int nnlist);
  double dist(const double r[]);

  double polar_prefactor(int, int, double);
  double associated_legendre(int, int, double);

  virtual void init_clebsch_gordan();
  double *cglist;                      // Clebsch-Gordan coeffs
  int idxcg_max;
  int chunksize;

  class ComputeOrientOrderAtom *c_orientorder;
  char *id_orientorder;
  double **qn_local;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Compute orientorder/atom requires a pair style be defined

Self-explanatory.

E: Compute orientorder/atom cutoff is longer than pairwise cutoff

Cannot compute order parameter beyond cutoff.

W: More than one compute orientorder/atom

It is not efficient to use compute orientorder/atom more than once.

*/
