make clean all
mpirun -np 2 ./karman -i test/initial.bin -o init.bin
./diffbin init.bin karman_ini.bin 

