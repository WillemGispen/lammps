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

ComputeStyle(angle/atom,ComputeAngleAtom)

#else

#ifndef LMP_COMPUTE_ANGLE_ATOM_H
#define LMP_COMPUTE_ANGLE_ATOM_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeAngleAtom : public Compute {
 public:
  ComputeAngleAtom(class LAMMPS *, int, char **);
  ~ComputeAngleAtom();
  virtual void init();
  void init_list(int, class NeighList *);
  virtual void compute_peratom();
  double memory_usage();
  double cutsq;
  int nnn;
  int bins;

  struct Sort {                     // data structure for sorting neighbors
    int nearest;                    // local ID of neighbor atom
    double distsq;                  // distance between center and neighbor atom
    double rlist[3];                // displacement between center and neighbor atom
  };

 protected:
  int nmax,maxneigh,ncol;
  class NeighList *list;
  double *distsq;
  int *nearest;
  double **rlist;
  double *alist;
  double **angle_array;

  void select3(int, int, double *, int *, double **);
  void calc_angle(double **rlist, int ncount, double angles[]);
  double dist(const double r[]);

  int idxcg_max;
  int chunksize;
  
  class ComputeVoronoi *c_voronoi;
  double **voro_local;
  // double **voro_atom;
  char *id_voronoi;

  Sort *sort;
  static int compare(const void *, const void *);
  static int compare_angle(const void *, const void *);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Compute angle/atom requires a pair style be defined

Self-explanatory.

E: Compute angle/atom cutoff is longer than pairwise cutoff

Cannot compute order parameter beyond cutoff.

W: More than one compute angle/atom

It is not efficient to use compute angle/atom more than once.

*/
