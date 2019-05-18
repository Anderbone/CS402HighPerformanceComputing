make clean all
export OMP_NUM_THREADS=2
mpirun -np 2 ./karman -i test/initial.bin -o init0.bin
gprof karman > init0.gprof
./diffbin init0.bin karman_ini.bin 

