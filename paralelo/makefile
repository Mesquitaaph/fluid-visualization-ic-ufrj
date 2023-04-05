default:

compile:
	nvcc -c nvrstks.cu

compile_windows: compile	
	nvcc -o paralel nvrstks.obj

compile_linux: compile	
	nvcc -o paralel nvrstks.o

run_windows: compile_windows
	./paralel ./input.txt ./output.txt