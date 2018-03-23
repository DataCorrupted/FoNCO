
/* ================================================
 * CUTEst interface to Python           
 *
 * Jiashan Wang UW
 *
 * ================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cutest.h"

void _open(char* fname, integer* funit)
{
	integer ierr = 0;
	FORTRAN_open(funit, fname, &ierr);
	return;
}

void _close(integer* funit)
{
	integer ierr = 0;
	FORTRAN_close(funit, &ierr);
	return;
}
void CUTEST_cdimen_1( integer *status, integer *funit, integer *n, integer *m )
{
	CUTEST_cdimen(status, funit, n, m);
	return;
}

void CUTEST_csetup_1( integer *status, integer *funit, integer *iout, 
             integer *io_buffer, integer *n, integer *m,
	      doublereal *x, doublereal *bl, doublereal *bu, 
              doublereal *v, doublereal *cl, doublereal *cu, 
	      logical *equatn, logical *linear, 
              integer *e_order, integer *l_order, integer *v_order )
{
	 CUTEST_csetup(status, funit, iout, io_buffer, n, m,
	               x, bl, bu, v, cl, cu, equatn, linear, 
				   e_order, l_order, v_order );
	 return;
}

void CUTEST_cfn_1( integer *status,  integer *n, integer *m, doublereal *x, 
          doublereal *f, doublereal *c )
{
	CUTEST_cfn(status, n, m, x, f, c );
	return;
}


void CUTEST_cofg_1( integer *status, integer *n, doublereal *x, doublereal *f, 
           doublereal *g, logical *grad )
{
	CUTEST_cofg( status, n, x, f, g, grad );
	return;
}

void CUTEST_cdimsj_1( integer *status, integer *nnzj )
{
	CUTEST_cdimsj(status, nnzj);
	return;
}

void CUTEST_ccfsg_1( integer *status,  integer *n, integer *m, doublereal *x, 
              doublereal *c, integer *nnzj, integer *lcjac,
	      doublereal *cjac, integer *indvar, integer *indfun,
	      logical *grad )
{
	CUTEST_ccfsg(status,  n, m, x, c, nnzj, lcjac,
	      cjac, indvar, indfun, grad);

	return;
}


void CUTEST_cshc_1( integer *status, integer *n, integer *m, doublereal *x, 
           doublereal *v, integer *nnzh, integer *lh, doublereal *h, 
           integer *irnh, integer *icnh )
{
	CUTEST_cshc(status, n, m, x, v, nnzh, lh, h, irnh, icnh );
	return;
}

void CUTEST_cish_1( integer *status, integer *n, doublereal *x, integer *iprob, 
            integer *nnzh,
	    integer *lh, doublereal *h, integer *irnh, integer *icnh )
{
	CUTEST_cish(status, n, x, iprob, nnzh, lh, h, irnh, icnh );
	return;

}


void CUTEST_cdimsh_1( integer *status, integer *nnzh )
{
	CUTEST_cdimsh( status, nnzh );
	return;
}



void CUTEST_cterminate_1( integer *status )
{
	CUTEST_cterminate(status);
	return;
}
