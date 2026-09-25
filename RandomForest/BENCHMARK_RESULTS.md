# Random Forest: Benchmarks de Desempenho e Energia

Avaliacao comparativa de treinamento, inferencia e consumo energetico entre as
implementacoes Sequencial (C11), OpenMP (16 threads) e CUDA (NVIDIA GeForce RTX 3050 Laptop GPU).

Ambiente experimental: CPU x86_64 instrumentada por Intel RAPL via Windows Performance
Data Helper (`pdh.dll`), GPU NVIDIA via NVIDIA Management Library (`nvml.dll`), Windows 11.
Potencia estatica de repouso registrada: 17.08 W. Repeticoes por medicao: 3.

Todos os numeros abaixo foram produzidos pela execucao que gerou este arquivo; nao ha
valores tabelados no codigo.

> Resolucao dos sensores: os contadores RAPL e NVML sao atualizados em intervalos da ordem
> de dezenas de milissegundos e incluem a energia de inicializacao do processo (contexto
> CUDA, carga do CSV). Execucoes muito curtas (abaixo de ~0,5 s, como `iris`) ficam no piso
> de resolucao e produzem potencias medias irrealistas; use `--repeats N --warmup` para
> medicoes energeticas comparaveis nesses casos.

---

## 1. Configuracao Experimental e Acuracia

Os hiperparametros de cada dataset foram selecionados por busca em grade com o binario
OpenMP (`python tune_params.py`), maximizando a acuracia media de teste sobre multiplas
particoes holdout. As tres implementacoes recebem exatamente os mesmos parametros e a
mesma semente, portanto treinam a mesma floresta.

| Dataset | N_Train | N_Test | Features | Classes | Trees | Max Depth | Min Samples | Mtry | Test Frac | Seed | Test Acc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iris | 120 | 30 | 4 | 3 | 300 | 12 | 5 | 2 | 0.2 | 42 | 0.9333 |
| breast-cancer | 455 | 114 | 30 | 2 | 300 | 6 | 5 | 3 | 0.2 | 42 | 0.9912 |
| sales_data | 800 | 200 | 13 | 4 | 200 | 10 | 5 | 8 | 0.2 | 42 | 0.2550 |
| letter-recog | 16000 | 4000 | 16 | 26 | 300 | 30 | 2 | 4 | 0.2 | 42 | 0.9617 |
| covtype | 464809 | 116203 | 54 | 7 | 50 | 30 | 2 | 18 | 0.2 | 42 | 0.9599 |

Sobre `sales_data`: o alvo `Product_Category` desse conjunto sintetico nao possui sinal
preditivo (4 classes, acuracia de teste no nivel do acaso mesmo com acuracia de treino
proxima de 1,0). Ele permanece na suite como caso de controle de sobreajuste, nao como
referencia de qualidade preditiva.

---

## 2. Desempenho Computacional

| Dataset | Seq Train (s) | OMP Train (s) | CUDA Train (s) | Speedup OMP Train | Speedup CUDA Train | Seq Pred (ms) | OMP Pred (ms) | CUDA Pred (ms) | CUDA Kern (ms) | Speedup OMP Pred | Speedup CUDA Pred | Speedup CUDA Kern |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iris | 0.0101 | 0.0038 | 0.0470 | 2.69x | 0.21x | 0.48 | 0.38 | 2.45 | 0.472 | 1.28x | 0.20x | 1.02x |
| breast-cancer | 0.1298 | 0.0216 | 0.0794 | 6.00x | 1.63x | 3.76 | 1.23 | 3.65 | 0.574 | 3.06x | 1.03x | 6.55x |
| sales_data | 0.6597 | 0.0645 | 0.4055 | 10.23x | 1.63x | 16.66 | 3.07 | 4.88 | 0.945 | 5.42x | 3.41x | 17.63x |
| letter-recog | 8.0747 | 1.0101 | 13.8373 | 7.99x | 0.58x | 1780.39 | 134.21 | 62.15 | 3.105 | 13.27x | 28.65x | 573.39x |
| covtype | 213.8062 | 34.4928 | 483.3195 | 6.20x | 0.44x | 13445.83 | 903.92 | 896.73 | 93.000 | 14.88x | 14.99x | 144.58x |

`CUDA Train (s)` usa o tempo de treino no dispositivo (`train_gpu_s`). `CUDA Kern (ms)`
isola o kernel de predicao, sem as transferencias host-device.

---

## 3. Consumo Energetico e Eficiencia (Energy-Delay Product)

| Dataset | Seq Energy (J) | OMP Energy (J) | CUDA Energy (J) | Seq Power (W) | OMP Power (W) | CUDA Power (W) | Seq EDP (J*s) | OMP EDP (J*s) | CUDA EDP (J*s) | Speedup OMP EDP | Speedup CUDA EDP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| iris | 8.16 | 3.45 | 10.55 | 350.72 | 201.02 | 68.90 | 0.19 | 0.06 | 1.62 | 3.21x | 0.12x |
| breast-cancer | 6.39 | 3.62 | 10.38 | 41.68 | 88.19 | 53.30 | 0.98 | 0.15 | 2.02 | 6.59x | 0.48x |
| sales_data | 22.81 | 8.13 | 27.58 | 38.11 | 96.76 | 52.64 | 13.79 | 0.68 | 14.45 | 20.20x | 0.95x |
| letter-recog | 161.87 | 41.91 | 559.67 | 16.28 | 34.01 | 35.84 | 1609.56 | 51.67 | 8740.69 | 31.15x | 0.18x |
| covtype | 3810.64 | 902.62 | 19297.19 | 16.28 | 21.28 | 38.18 | 891878.47 | 38293.32 | 9754199.05 | 23.29x | 0.09x |

### Descricao das Metricas Energeticas
- **Energia Total (Joules)**: integral da potencia ao longo da execucao ($E = E_{\text{CPU}} + E_{\text{GPU}}$), lida dos contadores de hardware em nanojoules (Intel RAPL) e milijoules (NVIDIA NVML).
- **Potencia Media (Watts)**: taxa media de dissipacao durante o processo ($\bar{P} = E_{\text{Total}} / \Delta t$).
- **Energy-Delay Product (EDP)**: produto entre energia e tempo ($\text{EDP} = E_{\text{Total}} \times \Delta t$). Penaliza solucoes lentas e prioriza o equilibrio entre consumo e produtividade.
- **Speedup de EDP**: ganho relativo frente a solucao sequencial ($\text{EDP}_{\text{Seq}} / \text{EDP}_{\text{Paralelo}}$).

---

## 4. Consistencia Numerica

| Dataset | Seq Test Acc | OMP Test Acc | CUDA Test Acc | CUDA Mismatches |
| --- | --- | --- | --- | --- |
| iris | 0.9333 | 0.9333 | 0.9333 | 0 |
| breast-cancer | 0.9912 | 0.9912 | 0.9912 | 0 |
| sales_data | 0.2550 | 0.2550 | 0.2450 | 0 |
| letter-recog | 0.9617 | 0.9617 | 0.9603 | 0 |
| covtype | 0.9599 | 0.9599 | 0.9603 | 0 |

As tres implementacoes recebem os mesmos parametros e a mesma semente de particionamento,
e a inferencia percorre a mesma travessia pre-ordem. Divergencias residuais de acuracia
podem ocorrer quando dois candidatos de corte empatam em ganho de Gini e a ordem de
avaliacao (sequencial na CPU, cooperativa na GPU) desfaz o empate de forma diferente.
`CUDA Mismatches` e a contagem de predicoes divergentes entre GPU e CPU para a mesma
floresta, reportada pelo proprio binario CUDA.

---

## 5. Arquitetura da Implementacao em GPU (CUDA)

O treinamento opera integralmente no dispositivo por meio de dois kernels:

1. **Memoria compartilhada (`rf_train_forest_kernel_small`)**: conjuntos cujas amostras
   cabem na SRAM do bloco ($N \le 1024$), como `iris`, `breast-cancer` e `sales_data`.
   Cada bloco (256 threads) treina uma arvore completa; as ordenacoes de pares
   (valor, classe) usam redes bitonicas cooperativas, sem trafego adicional para a VRAM.
2. **Memoria global com lotes dinamicos (`rf_train_forest_kernel`)**: para $N > 1024$,
   como `letter-recognition` e `covtype`. O espaco de trabalho em VRAM e fatiado em lotes
   de arvores (teto de 512 MB por lote) para evitar exaustao de memoria, com ordenacao
   bitonica em memoria global e reducoes paralelas de impureza de Gini.
3. **Consistencia**: uma pilha LIFO no dispositivo espelha a travessia pre-ordem da CPU
   (ver secao 4 para a contagem de divergencias medida).

---

## 6. Analise dos Resultados

1. **Treinamento em OpenMP**: speedup de 2.69x a 10.23x sobre a versao sequencial, com reducao de energia total de 1.77x a 4.22x e ganho de EDP de 3.21x a 31.15x. A potencia instantanea da CPU cresce sob carga total, mas a compressao do tempo de execucao mais que compensa esse aumento.
2. **Treinamento em CUDA**: speedup de 0.21x a 1.63x (2 de 5 datasets acima de 1,00x). A construcao de arvores e irregular e dependente de ordenacoes/reducoes sequenciais por no, o que limita a ocupacao da GPU; o ganho aparece apenas quando ha volume suficiente para saturar os multiprocessadores.
3. **Inferencia**: speedup de 1.28x a 14.88x em OpenMP e 0.20x a 28.65x em CUDA no tempo total de predicao, chegando a 1.02x a 573.39x quando se considera apenas o kernel (sem as transferencias host-device). A predicao e massivamente paralela e sem divergencia de trabalho, sendo o regime mais favoravel a GPU.
4. **Eficiencia energetica (EDP)**: OpenMP domina em todos os casos (3.21x a 31.15x); CUDA fica acima do sequencial em 0 de 5 datasets (0.09x a 0.95x), porque a GPU mantem potencia elevada durante todo o treinamento, penalizando o produto energia-atraso quando o kernel nao converte essa potencia em reducao de tempo.

---

## 7. Cenario de Alta Demanda (covtype.csv)

O conjunto `covtype.csv` (581.012 amostras, 54 atributos, 7 classes) e executado
com os mesmos binarios e a mesma instrumentacao dos demais, com arvores profundas
(`max_depth=30`, `mtry=18`, `trees=50`), regime em que a acuracia de
teste alcanca 95.99%.

| Implementacao | Treino (s) | Predicao Total (ms) | Kernel Predicao (ms) | Speedup Treino | Speedup Predicao | Energia (J) | EDP (J*s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Sequencial (C11) | 213.8062 | 13445.83 | N/A | 1.00x | 1.00x | 3810.64 | 891878.47 |
| OpenMP (16 threads) | 34.4928 | 903.92 | N/A | 6.20x | 14.88x | 902.62 | 38293.32 |
| CUDA (NVIDIA GeForce RTX 3050 Laptop GPU) | 483.3195 | 896.73 | 93.000 | 0.44x | 14.99x | 19297.19 | 9754199.05 |

Consistencia numerica: 0 divergencia(s) entre GPU e CPU nas 116203 amostras de teste.
