cdef extern from "quadtree.hpp" nogil:
	cdef cppclass quadArray:
		quadArray( double bmin[2], double bmax[2], int size )
		void build_tree( const double* particles )
		void compute_mass_distribution( const double* particles, const double* mass )
		void compute_force( const double* p, double* acc ) const

		int nbodies()
		int ncell  ()
		int* child()
		double* cell_center()
		double* cell_radius()
