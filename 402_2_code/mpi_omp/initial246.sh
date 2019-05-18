
mpirun -np 2 ./karman -i test/initial.bin -o out.bin 
gprof karman > i20.gprof
mpirun -np 4 ./karman -i test/initial.bin -o out.bin 
gprof karman > i40.gprof
mpirun -np 6 ./karman -i test/initial.bin -o out.bin 
gprof karman > i60.gprof
mpirun -np 2 ./karman -i test/initial.bin -o out.bin 
gprof karman > i21.gprof
mpirun -np 4 ./karman -i test/initial.bin -o out.bin 
gprof karman > i41.gprof
mpirun -np 6 ./karman -i test/initial.bin -o out.bin 
gprof karman > i61.gprof
mpirun -np 2 ./karman -i test/initial.bin -o out.bin 
gprof karman > i22.gprof
mpirun -np 4 ./karman -i test/initial.bin -o out.bin 
gprof karman > i42.gprof
mpirun -np 6 ./karman -i test/initial.bin -o out.bin 
gprof karman > i62.gprof
./diffbin out.bin karman_ini.bin
