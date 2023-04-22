default:

compile:
	nvcc -c nvrstks.cu

compile_original:
	nvcc -c nvrstks_original.cu

compile_windows: compile	
	nvcc -o paralel nvrstks.obj

compile_windows_original: compile_original
	nvcc -o paralel nvrstks_original.obj

compile_linux: compile	
	nvcc -o paralel nvrstks.o

run_windows: compile_windows
	./paralel ./input.txt ../estudo-visualizacao/mtxp2.txt

run_windows_original: compile_windows_original
	./paralel ./input.txt ../estudo-visualizacao/mtxp2.txt