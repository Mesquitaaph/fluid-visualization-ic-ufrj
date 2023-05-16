compilando glfw no windows:
gcc .\nvrstks.c -o .\nvrstks.exe -lopengl32 -lgdi32 -luser32 -lkernel32 -lglfw3 -lglfw3dll


otimização
gcc -march=native -ffast-math -O2 -ftree-vectorize .\nvrstks.c -o .\nvrstks.exe