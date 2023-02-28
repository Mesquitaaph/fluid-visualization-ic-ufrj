compilando nvrstks.cu no Ubuntu

nvcc -c nvrstks.cu
nvcc -o paralel nvrstks.o

com visualizacao
nvcc -c nvrstks.cu
nvcc -o paralel nvrstks.o -lglut -lGL -lGLU -lglfw


./paralel ./input.txt ./output.txt