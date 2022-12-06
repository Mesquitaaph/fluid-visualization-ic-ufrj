compilando glfw no windows:
gcc .\calor.c -o .\calor.exe -lopengl32 -lgdi32 -luser32 -lkernel32 -lglfw3 -lglfw3dll

verificar esse doc no ubuntu
http://geminiweb.slu.edu/wp-content/uploads/CUDA_Getting_Started_Linux.pdf
pag 20

compilando calor.cu no Ubuntu
nvcc -o calor_gpu calor.cu -lglut -lGL -lGLU -lglfw

nvcc -c solve-LU.cu "calor1d/calor.cu"
nvcc -o calor solve-LU.o "calor.o" -lglut -lGL -lGLU -lglfw