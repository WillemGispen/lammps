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

ComputeStyle(coarseorientorder/atom,ComputeCoarseOrientOrderAtom)

#else

#ifndef LMP_COMPUTE_COARSEORIENTORDER_ATOM_H
#define LMP_COMPUTE_COARSEORIENTORDER_ATOM_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeCoarseOrientOrderAtom : public Compute {
 public:
  ComputeCoarseOrientOrderAtom(class LAMMPS *, int, char **);
  ~ComputeCoarseOrientOrderAtom();
  virtual void init();
  void init_list(int, class NeighList *);
  virtual void compute_peratom();
  double memory_usage();
  double cutsq;
  int iqlcomp, qlcomp, qlcompflag, wlflag, wlhatflag;
  int *qlist;
  int nqlist;

 protected:
  int nmax,maxneigh,ncol,nnn;
  class NeighList *list;
  double *distsq;
  int *nearest;
  double **rlist;
  int qmax;
  double **qnarray;
  double **qnm_r;
  double **qnm_i;

  void select3(int, int, double *, int *, double **);
  void calc_boop(double **rlist, int numNeighbors,
                 double qn[], int nlist[], int nnlist);
  double dist(const double r[]);

  double polar_prefactor(int, int, double);
  double associated_legendre(int, int, double);

  static const int nmaxfactorial = 167;
  static const double nfac_table[];
  double factorial(int);
  virtual void init_clebsch_gordan();
  double *cglist;                      // Clebsch-Gordan coeffs
  int idxcg_max;
  int chunksize;
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Compute coarseorientorder/atom requires a pair style be defined

Self-explanatory.

E: Compute coarseorientorder/atom cutoff is longer than pairwise cutoff

Cannot compute order parameter beyond cutoff.

W: More than one compute coarseorientorder/atom

It is not efficient to use compute coarseorientorder/atom more than once.

*/
