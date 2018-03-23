[ -z "$CUTEST" ] && echo "Environment CUTEST not set. Have you installed CUTEst correctly?" && exit 1;
[ -z "$MYARCH" ] && echo "Environment MYARCH not set. Have you installed CUTEst correctly?" && exit 1;

cp $CUTEST/objects/$MYARCH/double/libcutest.a .
cp $CUTEST/modules . -rf
