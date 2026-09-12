## Análise de Escalabilidade com OpenMP

### Metodologia

Para avaliar a escalabilidade da implementação paralela com OpenMP, foi realizado um experimento de **strong scaling**, no qual o tamanho do problema foi mantido fixo enquanto o número de threads foi aumentado.

Foi utilizado o dataset `covtype.csv`, contendo 581.012 amostras, 54 atributos e 7 classes. O conjunto foi dividido em 464.809 amostras para treinamento e 116.203 para teste.

Os parâmetros utilizados no Random Forest foram:

- Número de árvores: 100
- Profundidade máxima: 8
- `mtry`: 4
- Número mínimo de amostras para divisão: 2
- Seed: 42

Foram testadas as seguintes quantidades de threads:

`1, 2, 4, 8, 12 e 16`

Para cada configuração foram realizadas 5 execuções. Como medida representativa do tempo de execução, foi utilizada a **mediana** das cinco execuções, reduzindo a influência de variações ocasionais causadas pelo sistema operacional ou por outros processos em execução.

O speedup é calculado por:

$$S(p) = \frac{T_1}{T_p}$$

em que:
- $T_1$ é o tempo com uma thread;
- $T_p$ é o tempo com $p$ threads.

A eficiência paralela foi calculada por:

$$E(p) = \frac{S(p)}{p}$$

em que:
- $S(p)$ representa o speedup obtido com $p$ threads;
- $p$ representa o número de threads utilizadas.

---

### Resultados do Treinamento

| Threads | Tempo mediano (s) | Speedup | Eficiência |
|--------:|------------------:|--------:|-----------:|
| 1  | 101.169 | 1.00x | 100.0% |
| 2  | 51.106  | 1.98x | 99.0% |
| 4  | 27.308  | 3.70x | 92.6% |
| 8  | 15.065  | 6.72x | 83.9% |
| 12 | 12.343  | 8.20x | 68.3% |
| 16 | 11.144  | 9.08x | 56.7% |

Os resultados mostram boa escalabilidade principalmente entre 1 e 8 threads. Com 2 threads, o speedup obtido foi de aproximadamente 1,98x, bastante próximo do valor ideal de 2x. Com 4 threads, o speedup chegou a 3,70x, mantendo eficiência superior a 90%.

Com 8 threads, o treinamento apresentou speedup de aproximadamente 6,72x e eficiência de 83,9%. A partir desse ponto, entretanto, os ganhos passam a apresentar retornos decrescentes. Com 12 threads, o speedup foi de 8,20x e, com 16 threads, de 9,08x.

Embora a quantidade de threads tenha dobrado de 8 para 16, o tempo de treinamento caiu de aproximadamente 15,07 segundos para 11,14 segundos. Isso indica que o programa começa a atingir uma região de saturação, na qual o aumento do paralelismo deixa de produzir ganhos proporcionais.

Essa redução de eficiência pode ser explicada por fatores como overhead do OpenMP, contenção no acesso à memória, compartilhamento de cache, largura de banda da memória e partes do algoritmo que não podem ser paralelizadas.

---

### Resultados da Predição

| Threads | Tempo mediano (s) | Speedup | Eficiência |
|--------:|------------------:|--------:|-----------:|
| 1  | 1.452 | 1.00x | 100.0% |
| 2  | 0.737 | 1.97x | 98.4% |
| 4  | 0.398 | 3.65x | 91.2% |
| 8  | 0.216 | 6.71x | 83.9% |
| 12 | 0.157 | 9.22x | 76.9% |
| 16 | 0.130 | 11.19x | 69.9% |

A etapa de predição também apresentou boa escalabilidade. Com 16 threads, o tempo caiu de aproximadamente 1,45 segundos para 0,13 segundos, correspondendo a um speedup de aproximadamente 11,19x.

A predição apresentou eficiência maior que o treinamento para 12 e 16 threads. Isso ocorre porque as amostras podem ser avaliadas de forma independente, possibilitando uma distribuição mais uniforme do trabalho entre as threads.

---

### Correção dos Resultados

Além do desempenho, foi verificado se o aumento do número de threads alterava o resultado do algoritmo.

Em todas as configurações foram obtidos os mesmos valores:

- Acurácia de treinamento: `0.6288` (62,88%)
- Acurácia de teste: `0.6297` (62,97%)

Isso indica que a paralelização não alterou o comportamento numérico do Random Forest, afetando apenas seu tempo de execução.

---

### Análise da Escalabilidade

O comportamento observado é característico de aplicações paralelas com **strong scaling**. Para números pequenos de threads, o ganho é próximo ao ideal, porém a eficiência diminui conforme mais recursos computacionais são adicionados.

No treinamento, o algoritmo apresentou eficiência próxima de 99% com duas threads e superior a 83% com oito threads. Entretanto, com 16 threads, a eficiência caiu para aproximadamente 56,7%.

Essa queda não significa que a implementação deixou de se beneficiar do paralelismo. O tempo caiu de aproximadamente 101 segundos para apenas 11 segundos, representando uma aceleração superior a 9 vezes. Entretanto, a partir de aproximadamente 8 threads, cada nova thread adicionada proporciona um ganho proporcionalmente menor.

A paralelização do treinamento ocorre no nível das árvores do Random Forest. Como cada árvore pode possuir tempos de construção diferentes, é utilizado escalonamento dinâmico (`schedule(dynamic)`), permitindo que uma thread que finalize sua árvore receba uma nova tarefa disponível.

Na etapa de predição, as amostras são independentes entre si e são distribuídas entre as threads utilizando escalonamento estático. Esse tipo de processamento apresenta maior regularidade, contribuindo para uma eficiência superior nas configurações com maior quantidade de threads.

---

### Conclusão

Os experimentos demonstram que a implementação com OpenMP apresenta ganho significativo de desempenho em relação à execução com uma única thread.

Para o treinamento, o melhor tempo observado entre as configurações testadas foi de aproximadamente 11,14 segundos utilizando 16 threads, contra aproximadamente 101,17 segundos utilizando apenas uma thread, resultando em speedup de 9,08x.

Na predição, o speedup alcançou aproximadamente 11,19x com 16 threads.

Os resultados também mostram que o ganho de desempenho não cresce linearmente com a quantidade de threads. A partir de aproximadamente 8 threads, ocorre redução progressiva da eficiência paralela, indicando a presença de limitações relacionadas ao overhead de paralelização e aos recursos compartilhados do processador.

Mesmo assim, a utilização do OpenMP proporcionou uma redução expressiva no tempo de execução, mantendo exatamente a mesma acurácia em todas as configurações avaliadas.