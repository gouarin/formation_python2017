#ifndef _NBODY_FORCES_HPP_
#define _NBODY_FORCES_HPP_
#include <cmath>

namespace {

    const double eps      = 0.01; // Approx. 3 light year
    const double gamma_si = 6.67408e-11;

    void force( const double *p1, const double *p2, double m2, double *ret ) {
        double dx = p2[0] - p1[0];
        double dy = p2[1] - p1[1];

        double dist = sqrt( dx * dx + dy * dy + eps );

        double f = 0.;
        if ( dist > 0 ) {
            f = ( gamma_si * m2 ) / ( dist * dist * dist );
        }
        ret[0] = f * dx;
        ret[1] = f * dy;
    }
}

#endif