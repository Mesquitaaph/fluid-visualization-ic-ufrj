default:

compile:
	nvcc -O3 -c nvrstks.cu

compile_original:
	nvcc -O3 -c nvrstks_original.cu

compile_visu:
	nvcc -O3 -c nvrstks_visu.cu

compile_windows: compile	
	nvcc -o paralel nvrstks.obj

compile_windows_original: compile_original
	nvcc -o paralel nvrstks_original.obj

compile_linux: compile	
	nvcc -o paralel nvrstks.o

compile_linux_visu: compile_visu
	nvcc -o paralel nvrstks_visu.o -lglut -lGL -lGLU -lglfw

run_windows: compile_windows
	./paralel ./input.txt ../estudo-visualizacao/mtxp2.txt

run_windows_original: compile_windows_original
	./paralel ./input.txt ../estudo-visualizacao/mtxp2.txt

run_linux: compile_linux
	./paralel ./input.txt ../estudo-visualizacao/mtxp2.txt

run_linux_visu: compile_linux_visu
	./paralel ./input.txt ../estudo-visualizacao/mtxp2.txt