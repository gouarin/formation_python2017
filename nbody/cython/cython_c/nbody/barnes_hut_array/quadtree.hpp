#ifndef _BARNES_HUT_ARRAY_QUADTREE_HPP
#define _BARNES_HUT_ARRAY_QUADTREE_HPP
#include <cmath>
#include <vector>

class quadArray {
public:
    quadArray( double bmin[2], double bmax[2], int size );
    void build_tree( const double *particles );
    void compute_mass_distribution( const double *particles,
                                    const double *mass );
    void compute_force( const double *p, double *acc ) const;

    int           nbodies( ) const { return m_nbodies; }
    int           ncell( ) const { return m_ncell; }
    const int *   child( ) const { return &m_child[0]; }
    int *         child( ) { return &m_child[0]; }
    const double *cell_center( ) const { return &m_cell_center[0]; }
    double *      cell_center( ) { return &m_cell_center[0]; }
    const double *cell_radius( ) const { return &m_cell_radius[0]; }
    double *      cell_radius( ) { return &m_cell_radius[0]; }

private:
    int                 m_nbodies, m_ncell;
    std::vector<int>    m_child;
    double              m_bmin[2], m_bmax[2], m_center[2], m_box_size[2];
    std::vector<double> m_cell_center, m_cell_radius, m_mass, m_center_of_mass;
};

#endif
