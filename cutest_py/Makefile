
FORTRAN = /usr/bin/gfortran
BASIC = -fno-second-underscore -shared 
CCBASIC   = -c -ansi -pedantic -fPIC
CC        = gcc
LIBDIR = -L./
LIBFLAG = -lcutest -lgfortran
INDIR = -I./
OBJ = py_cute
PROBFLAG = -I./modules/pc64.lnx.gf5/double -J./modules/pc64.lnx.gf5/double -c -fno-second-underscore -O -ffixed-form -fPIC

all: ${OBJDIR}/$(OBJ).so clean

EXTER.o: ${OBJDIR}/EXTER.f
	${FORTRAN} ${PROBFLAG} -o EXTER.o ${OBJDIR}/EXTER.f

ELFUN.o: ${OBJDIR}/ELFUN.f
	${FORTRAN} ${PROBFLAG} -o ELFUN.o ${OBJDIR}/ELFUN.f

GROUP.o: ${OBJDIR}/GROUP.f
	${FORTRAN} ${PROBFLAG} -o GROUP.o ${OBJDIR}/GROUP.f

RANGE.o: ${OBJDIR}/RANGE.f
	${FORTRAN} ${PROBFLAG} -o RANGE.o ${OBJDIR}/RANGE.f


${OBJ}.o: ${OBJ}.c
	${CC} ${CCBASIC} ${OBJ}.c -o ${OBJ}.o $(INDIR)

${OBJDIR}/${OBJ}.so: ${OBJ}.o ELFUN.o GROUP.o RANGE.o EXTER.o 
	${FORTRAN} ${BASIC} -o ${OBJDIR}/${OBJ}.so ELFUN.o GROUP.o RANGE.o EXTER.o ${OBJ}.o ${LIBDIR} ${LIBFLAG} ${INDIR}


clean:
	rm -f *.o
#gcc -c -ansi -pedantic -o cute_test.o cute_test.c  -I/Users/jsw1119/test_problem/cutest/include
#gcc -c -ansi -pedantic -o genc.o genc.c  -I/Users/jsw1119/test_problem/cutest/include
#gcc -c -ansi -pedantic -o genc_main.o genc_main.c  -I/Users/jsw1119/test_problem/cutest/include
#
#gfortran -fno-second-underscore -flat_namespace  -o cute_test ELFUN.o GROUP.o RANGE.o cute_test.o -L/Users/jsw1119/test_problem/cutest/objects/pc64.lnx.gf5/double -lcutest
#gfortran -fno-second-underscore -flat_namespace  -shared -o run_genc.so ELFUN.o GROUP.o RANGE.o genc.o genc_main.o -L/Users/jsw1119/test_problem/cutest/objects/pc64.lnx.gf5/double  -lcutest
