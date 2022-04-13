# fluid-visualization-ic-ufrj

Nesta apresentação haverá um resumo sobre as três formas principais encontradas de criar aplicações com CUDA: CUDA C, nvcc4jupyter e PyCUDA e uma breve reflexão sobre qual ou quais escolher, pensando na praticidade, na performance e conforto do grupo.

## CUDA C
Necessita uma placa de vídeo instalada no computador com compatibilidade com a tecnologia CUDA, ou seja, placa NVIDIA.

Sobre a sua linguagem: foi percebido até então que é extremamente similar às linguagens C/C++, com a diferença que não necessita importar uma biblioteca específica para executar as principais funções da CUDA. Outras diferenças são a extensão do arquivo ".cu" e o uso do qualificador "\_\_global\_\_", que caracteriza uma função como um kernel do programa. Essas semelhanças não são surpresas, já que é, na verdade, uma extensão da linguagem C.

Sobre a performance é de se esperar uma performance tão boa quanto melhor for a configuração do computador que for rodar a aplicação e o algoritmo utilizado, justamente por ser compilado em C.

Sobre a praticidade e conforto é esperado escrever mais que em PyCuda, se preocupando mais com tipagem e gerenciamento de memória.

Exemplo de código no arquivo "cuda.cu" neste repositório soma elementos de dois vetores A e B num vetor C e verifica, ao final, sua corretude.

## nvcc4jupyter
É um plugin utilizado no Google Colab com Python para executar programas escritos em CUDA C dentro de uma célula. Escreve-se literalmente o mesmo código, necessitando somente o indicador "%%cu" no início da célula.

Sobre a performance é esperado algo homogêneo entre diferentes computadores já que a GPU com CUDA disponível no Colab é ofertada pelo próprio serviço em nuvem da Google.

A praticidade e conforto parecem ser quase as mesmas de CUDA C, com a diferença que o Colab não reconhece na sintaxe do código apenas o que é comum ao Python.

Exemplo de código no arquivo "nvcc_plugin_exemplo.ipynb" neste repositório soma elementos de dois vetores A e B num vetor C e verifica, ao final, sua corretude.

## PyCuda no Colab
O PyCUDA é uma biblioteca que permite acessar a API de computação paralela CUDA da Nvidia a partir do Python. Sua camada base é escrita em C++, então todas as suas conveniências não possuem custo.

Sobre a performance, temos a mesma situação do "nvcc4jupyter", pois também é executado no Colab. 

A praticidade e conforto aqui podem ser mais atrativos no geral. O PyCUDA, por conta de ser em Python, apresenta uma sintaxe mais simples e, além disso, também possui algumas conveniências, tais como:
  - Gerenciamento de memória automático
  - Verificação automática de erros. Todos os erros CUDA são traduzidos em exeções do Python

Exemplo de código no arquivo "pycuda_exemplo.ipynb" neste repositório soma elementos de dois vetores A e B num vetor C e verifica, ao final, sua corretude.

## Reflexões
CUDA C possui garantia no quesito documentação (além da experiência da prof. Silvana) e compatibilidade com CUDA, justamente por ser próprio da NVIDIA. No entando, nem todos do grupo possuem placa de vídeo com CUDA.

A opção de nvcc4jupyter faz não ser necessário uma placa de vídeo com CUDA porém possui performance limitada e é mais dificil de identificar erros no código, por conta do Python não reconhecer a sintaxe de C.

Por fim a opção do PyCuda também faz não ser necessário uma placa de vídeo com CUDA porém com performance limitada, mas possivelmente essa limitação acaba se executar numa máquina com CUDA em vez do Colab. Já a questão da documentação: ela existe³ e parece suficiente, porém não sabemos muito sobre o que vamos utilizar para confirmar que é suficiente.

- Apostila de programação paralela em GPU com CUDA¹: http://arquivo.sbmac.org.br/arquivos/notas/livro_84.pdf
- Bom tutorial para rodar CUDA no Colab com o plugin **nvcc4jupyter**²: https://vitalitylearning.medium.com/running-cuda-in-google-colab-525a92efcf75
