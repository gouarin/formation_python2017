#include <algorithm>
#include <iostream>

#include "../forces.hpp"
#include "quadtree.hpp"

quadArray::quadArray( double bmin[2], double bmax[2], int size )
    : m_nbodies( size ), m_ncell( 0 ), m_child( 4 * ( 2 * size + 1 ), -1 ),
      m_cell_center( 2 * ( 2 * size + 1 ), 0. ),
      m_cell_radius( 2 * ( 2 * size + 1 ), 0. ), m_mass( ),
      m_center_of_mass( ) {
    m_bmin[0]        = bmin[0];
    m_bmin[1]        = bmin[1];
    m_bmax[0]        = bmax[0];
    m_bmax[1]        = bmax[1];
    m_center[0]      = 0.5 * ( bmin[0] + bmax[0] );
    m_center[1]      = 0.5 * ( bmin[1] + bmax[1] );
    m_box_size[0]    = bmax[0] - bmin[0];
    m_box_size[1]    = bmax[1] - bmin[1];
    m_cell_center[0] = m_center[0];
    m_cell_center[1] = m_center[1];
    m_cell_radius[0] = m_box_size[0];
    m_cell_radius[1] = m_box_size[1];
}
// ------------------------------------------------------------------------------------------
void quadArray::build_tree( const double *particles ) {
    int    cell, childPath, oldchildPath, newchildPath, childIndex, npart;
    double x, y;
    double center[2], box_size[2];
    m_ncell = 0;
    for ( int ip = 0; ip < m_nbodies; ++ip ) {
        center[0]   = m_center[0];
        center[1]   = m_center[1];
        box_size[0] = m_box_size[0];
        box_size[1] = m_box_size[1];
        x           = particles[4 * ip + 0];
        y           = particles[4 * ip + 1];
        cell        = 0;

        childPath = 0;
        if ( x > center[0] )
            childPath += 1;
        if ( y > center[1] )
            childPath += 2;

        childIndex = m_nbodies + childPath;

        while ( m_child[childIndex] > m_nbodies ) {
            cell      = m_child[childIndex] - m_nbodies;
            center[0] = m_cell_center[2 * cell + 0];
            center[1] = m_cell_center[2 * cell + 1];
            childPath = 0;
            if ( x > center[0] )
                childPath += 1;
            if ( y > center[1] )
                childPath += 2;
            childIndex = m_nbodies + 4 * cell + childPath;
        }
        // No particle on this cell, just add it
        if ( m_child[childIndex] == -1 ) {
            m_child[childIndex] = ip;
            m_child[ip]         = cell;
        }
        // this cell already has a particle
        // subdivide and set the two particles
        else if ( m_child[childIndex] < m_nbodies ) {
            npart = m_child[childIndex];

            oldchildPath = newchildPath = childPath;
            while ( oldchildPath == newchildPath ) {
                m_ncell += 1;
                m_child[childIndex] = m_nbodies + m_ncell;
                center[0]           = m_cell_center[2 * cell + 0];
                center[1]           = m_cell_center[2 * cell + 1];
                box_size[0]         = .5 * m_cell_radius[2 * cell + 0];
                box_size[1]         = .5 * m_cell_radius[2 * cell + 1];
                if ( oldchildPath & 1 )
                    center[0] += box_size[0];
                else
                    center[0] -= box_size[0];
                if ( ( oldchildPath >> 1 ) & 1 )
                    center[1] += box_size[1];
                else
                    center[1] -= box_size[1];

                oldchildPath = 0;
                if ( particles[4 * npart + 0] > center[0] )
                    oldchildPath += 1;
                if ( particles[4 * npart + 1] > center[1] )
                    oldchildPath += 2;

                newchildPath = 0;
                if ( particles[4 * ip + 0] > center[0] )
                    newchildPath += 1;
                if ( particles[4 * ip + 1] > center[1] )
                    newchildPath += 2;

                cell = m_ncell;

                m_cell_center[2 * m_ncell + 0] = center[0];
                m_cell_center[2 * m_ncell + 1] = center[1];
                m_cell_radius[2 * m_ncell + 0] = box_size[0];
                m_cell_radius[2 * m_ncell + 1] = box_size[1];

                childIndex = m_nbodies + 4 * m_ncell + oldchildPath;
            }
            m_child[childIndex] = npart;
            m_child[npart]      = m_ncell;

            childIndex          = m_nbodies + 4 * m_ncell + newchildPath;
            m_child[childIndex] = ip;
            m_child[ip]         = m_ncell;
        }
    }
}
// ------------------------------------------------------------------------------------------
void quadArray::compute_mass_distribution( const double *particles,
                                           const double *mass ) {
    std::vector<double>( m_nbodies + m_ncell + 1, 0. ).swap( m_mass );
    std::copy( mass, mass + m_nbodies, &m_mass[0] );
    std::vector<double>( 2 * ( m_nbodies + m_ncell + 1 ), 0. )
        .swap( m_center_of_mass );
    for ( int ip = 0; ip < m_nbodies; ++ip ) {
        m_center_of_mass[2 * ip + 0] = particles[4 * ip + 0];
        m_center_of_mass[2 * ip + 1] = particles[4 * ip + 1];
    }
    for ( int i = m_ncell; i >= 0; --i ) {
        int indi = m_nbodies + i;
        for ( int j = m_nbodies + 4 * i; j < m_nbodies + 4 * i + 4; ++j ) {
            int indj = m_child[j];
            if ( indj >= 0 ) {
                m_mass[indi] += m_mass[indj];
                m_center_of_mass[2 * indi + 0] +=
                    m_center_of_mass[2 * indj + 0] * m_mass[indj];
                m_center_of_mass[2 * indi + 1] +=
                    m_center_of_mass[2 * indj + 1] * m_mass[indj];
            }
        }
        m_center_of_mass[2 * indi + 0] /= m_mass[indi];
        m_center_of_mass[2 * indi + 1] /= m_mass[indi];
    }
    // for(std::size_t i=0; i<m_mass.size(); ++i)
    //     std::cout << m_mass[i] << " ";
    // std::cout << "\n";

    // for(std::size_t i=0; i<m_center_of_mass.size(); i+=2)
    //     std::cout << m_center_of_mass[i] << " " << m_center_of_mass[i+1] << "\n";
    // std::cout << "\n";
}
// ------------------------------------------------------------------------------------------
void quadArray::compute_force( const double *p, double *acc ) const {
    int              depth = 0;
    std::vector<int> localPos( 2 * m_nbodies, 0 );
    std::vector<int> localNode( 2 * m_nbodies, 0 );
    localNode[0] = m_nbodies;
    double pos[2], F[2];
    pos[0] = p[0];
    pos[1] = p[1];
    acc[0] = 0.;
    acc[1] = 0.;

    while ( depth >= 0 ) {
        while ( localPos[depth] < 4 ) {
            int child = m_child[localNode[depth] + localPos[depth]];
            localPos[depth] += 1;
            if ( child >= 0 ) {
                if ( child < m_nbodies ) {
                    force( pos, &m_center_of_mass[2 * child], m_mass[child],
                           F );
                    acc[0] += F[0];
                    acc[1] += F[1];
                } else {
                    double dx   = m_center_of_mass[2 * child + 0] - pos[0];
                    double dy   = m_center_of_mass[2 * child + 1] - pos[1];
                    double dist = sqrt( dx * dx + dy * dy );
                    if ( ( dist != 0 ) &&
                         ( m_cell_radius[2 * ( child - m_nbodies )] / dist <
                           .5 ) ) {
                        force( pos, &m_center_of_mass[2 * child], m_mass[child],
                               F );
                        acc[0] += F[0];
                        acc[1] += F[1];
                    } else {
                        depth += 1;
                        localNode[depth] =
                            m_nbodies + 4 * ( child - m_nbodies );
                        localPos[depth] = 0;
                    }
                }
            }
        }
        depth -= 1;
    }
}
