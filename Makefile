sobel: main.o libsobel_cuda.so libsobel.so
	g++-7 -std=c++11 $^ -O2 -o $@ -L. -lsobel -lsobel_cuda -lboost_system -lboost_filesystem `pkg-config --libs opencv` 

main.o: main.cpp
	g++-7 -std=c++11 -O2 -c $^  -o $@

libsobel_cuda.so: sobel.cu
	nvcc -std=c++11 $^ -O2 -D_FORCE_INLINES -Xcompiler -fPIC -shared  -o $@

libsobel.so: sobel.cpp
	g++-7 -std=c++11 $^ -O2 -fPIC -shared -Wl,-soname,libsobel.so -o $@

clean: 
	rm -f sobel main.o libsobel_cuda.so libsobel.so