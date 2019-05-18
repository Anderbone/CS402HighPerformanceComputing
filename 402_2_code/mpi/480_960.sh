
mpirun -np 4 ./karman -x 480 -i test/480.bin -o out.bin
gprof karman > 480.gprof
mpirun -np 4 ./karman -x 960 -i test/960.bin -o out.bin
gprof karman > 960.gprof
