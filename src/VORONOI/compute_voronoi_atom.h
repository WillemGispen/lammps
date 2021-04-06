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

ComputeStyle(voronoi/atom,ComputeVoronoi)

#else

#ifndef LMP_COMPUTE_VORONOI_H
#define LMP_COMPUTE_VORONOI_H

#include "compute.h"

namespace voro {
  class container;
  class container_poly;
  class voronoicell_neighbor;
}

namespace LAMMPS_NS {

class ComputeVoronoi : public Compute {
 public:
  ComputeVoronoi(class LAMMPS *, int, char **);
  ~ComputeVoronoi();
  void init();
  void compute_peratom();
  void compute_vector();
  void compute_local();
  double memory_usage();
  int faces_flag;
  int sig_flag;

  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);

 private:
  voro::container *con_mono;
  voro::container_poly *con_poly;

  void buildCells();
  void checkOccupation();
  void loopCells();
  void processCell(voro::voronoicell_neighbor&, int);

  int nmax, rmax, maxedge, sgroupbit;
  char *radstr;
  double fthresh, ethresh;
  double **voro;
  double *edge, *sendvector, *rfield;
  enum { VOROSURF_NONE, VOROSURF_ALL, VOROSURF_GROUP } surface;
  bool onlyGroup, occupation;

  tagint *tags, oldmaxtag;
  int *occvec, *sendocc, *lroot, *lnext, lmax, oldnatoms, oldnall;
  int nfaces, nfacesmax;
  double **faces;
  
  void minkowski_tensor_invariants(std::vector<double>, std::vector<double>,
                                   std::vector<double>, std::vector<double>, double *);
  void minkowski_scalars(double *);
  // void minkowski_w210(double *);
  // void minkowski_w110(double *);
  // void minkowski_w220(double *);
  // void minkowski_w102(double *);
  void minkowski_w202(std::vector<double>, std::vector<double>, double *);
  void minkowski_w204(std::vector<double>, std::vector<double>, double *);

  double cutsq;
  void select3(int, int, double *, double *);
  static int compare_area(const void *, const void *);
  static int compare_dist(const void *, const void *);
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Could not find compute/voronoi surface group ID

Self-explanatory.

E: Illegal compute voronoi/atom command (occupation and (surface or edges))

Self-explanatory.

E: Compute voronoi/atom occupation requires an atom map, see atom_modify

UNDOCUMENTED

E: Compute voronoi/atom occupation requires atom IDs

UNDOCUMENTED

E: Variable name for voronoi radius does not exist

Self-explanatory.

E: Variable for voronoi radius is not atom style

Self-explanatory.

E: Voro++ error: narea and neigh have a different size

This error is returned by the Voro++ library.

*/
