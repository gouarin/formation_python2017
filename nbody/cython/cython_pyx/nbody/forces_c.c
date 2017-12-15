#include "forces_c.h"
#include <math.h>

void force( double epsi, double g_si, double *p1, double *p2, double m2,
            double *ret ) {
    double dx, dy;
    double dist;
    double F;
    dx   = p2[0] - p1[0];
    dy   = p2[1] - p1[1];
    dist = sqrt( dx * *2 + dy * *2 + epsi ) F = 0.;
    if ( dist > 0 )
        F  = ( g_si * m2 ) / ( dist * dist * dist );
    ret[0] = F * dx;
    ret[1] = F * dy;
}
