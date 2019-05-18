
make clean all
export OMP_NUM_THREADS=2
mpirun -np 2 ./karman -i test/initial.bin -o out.bin
gprof karman > out/20.gprof
./diffbin out.bin karman_ini.bin 
mpirun -np 2 ./karman -i test/initial.bin -o out.bin
gprof karman > out/21.gprof
mpirun -np 2 ./karman -i test/initial.bin -o out.bin
gprof karman > out/22.gprof




