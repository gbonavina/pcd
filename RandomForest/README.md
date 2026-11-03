# Random Forest em OpenMP

Código, dados e resultados do artigo *Explorando o Paralelismo em Random Forests:
Desempenho, Fidelidade Preditiva e Eficiência Energética com OpenMP*
(`report/template/artigo.pdf`).

O artigo mede o escalonamento forte de uma implementação OpenMP do Random Forest
(CART, Gini, bagging e subespaço aleatório de atributos) em quatro bases, de 1 a
32 threads, com tempo de treino e predição, acurácia de teste e energia do pacote
da CPU lida pelo Intel RAPL. Cada configuração é repetida em ordem embaralhada, e
os resultados trazem média, mediana e desvio padrão. As medições foram feitas num
Intel Core i5-12500H (4 núcleos P e 8 núcleos E, 16 threads lógicas) sob Windows 11.

## Estrutura

```text
RandomForest/
├── openmp/                     Implementação OpenMP usada no artigo
│   ├── Makefile
│   └── src/                    main.c, rf.c, rf.h
├── data/                       Bases do artigo
│   ├── iris.csv
│   ├── breast-cancer.csv       (WDBC)
│   ├── letter-recognition.data
│   └── covtype.csv
├── python/
│   ├── measure_energy.py       Leitura do RAPL (Windows PDH) usada pela varredura
│   ├── tune_params.py          Busca em grade dos hiperparâmetros
│   ├── sweep_small.csv         Resultado da busca: Iris e WDBC (inclui sales_data, fora do artigo)
│   └── sweep_letter.csv        Resultado da busca: Letter
├── run_openmp_experiments.py   Varredura 1, 2, 4, 8, 12, 16 e 32 threads com energia
├── run_openmp_experiments.sh   Atalho (bash)
├── run_openmp_experiments.bat  Atalho (Windows)
├── openmp_runs.csv             Dados brutos: uma linha por execução (bloco, posição, threads)
├── openmp_scaling.csv          Agregado (média, mediana, dp, speedups): Tabelas 2 a 4 e Figuras 1 e 2
├── report/
│   ├── LICENSE                 Licença do template SBC
│   └── template/
│       ├── artigo.tex          Fonte do artigo
│       ├── artigo.bib          Referências
│       ├── artigo.pdf          Artigo compilado
│       ├── sbc-template.sty, sbc.bst, caption2.sty   Estilo SBC
│       └── figuras/
│           ├── speedup.tex/.pdf       Figura 1 (pgfplots)
│           ├── edp.tex/.pdf           Figura 2 (pgfplots)
│           └── arvore_iris.dot/.pdf   Árvore ilustrativa (fora do artigo)
├── Makefile                    Atalhos para compilar, rodar e gerar o PDF
└── nao_utilizados/             Material que não entrou no artigo final
```

A pasta `nao_utilizados/` guarda o que foi desenvolvido no projeto mas não aparece
no artigo: as versões sequencial (`src/`) e CUDA (`cuda/`), os benchmarks
comparativos Seq/OpenMP/CUDA, a base `sales_data.csv`, saídas avulsas de árvores
em DOT e os arquivos de exemplo do template SBC. Veja `nao_utilizados/LEIA-ME.md`.

## Reprodução

Pré-requisitos: GCC com OpenMP, Python 3 e, para a energia, Windows com o contador
`Energy Meter` (RAPL) disponível no PDH. Para o PDF, uma distribuição LaTeX com
`latexmk` e `pgfplots`.

```bash
# 1. Compilar o binário OpenMP
make -C openmp

# 2. Varredura de escalonamento forte (gera openmp_runs.csv e openmp_scaling.csv; cerca de 1h30)
python run_openmp_experiments.py --overwrite
#    Recalcular só o agregado a partir dos dados brutos:
python run_openmp_experiments.py --aggregate-only

# 3. (Opcional) Busca em grade dos hiperparâmetros
python python/tune_params.py --threads 16 --export-csv python/sweep.csv

# 4. Compilar o artigo
cd report/template/figuras && lualatex speedup.tex && lualatex edp.tex
cd .. && latexmk -lualatex artigo.tex
```

Protocolo da varredura: cada base roda em blocos (30 em Iris e WDBC, 10 no Letter,
5 no Covertype); cada bloco executa uma vez cada número de threads, em ordem
embaralhada com `--order-seed` (padrão 20261103). Há uma execução de aquecimento
descartada por base, pausa de 1 s entre execuções e nenhum controle de afinidade
(o libgomp do MinGW não suporta `OMP_PLACES`/`OMP_PROC_BIND`). Sem `--append` ou
`--overwrite`, o script se recusa a sobrescrever um `openmp_runs.csv` existente.
O contador PDH do RAPL acumula picowatt-hora; `python/measure_energy.py` converte
para joules.

O `Makefile` da raiz expõe os mesmos passos: `make`, `make experimentos`,
`make tune` e `make artigo`.

Hiperparâmetros usados (semente 42, holdout de 20%):

| Base     | Árvores | Profundidade | `min_samples` | `mtry` |
| :------- | ------: | -----------: | ------------: | -----: |
| Iris     | 300     | 12           | 5             | 2      |
| WDBC     | 300     | 6            | 5             | 3      |
| Letter   | 300     | 30           | 2             | 4      |
| Covertype| 50      | 30           | 2             | 18     |

### Argumentos do binário

```text
  --data PATH         Arquivo CSV de dados (padrão: data/iris.csv)
  --target COL        Coluna alvo, por nome ou índice (padrão: última coluna)
  --trees N           Número de árvores (padrão: 100)
  --max-depth D       Profundidade máxima (padrão: 8)
  --min-samples M     Mínimo de amostras para dividir um nó (padrão: 2)
  --mtry K            Atributos sorteados por nó (0 = floor(sqrt(d)))
  --test-frac F       Fração de teste do holdout (padrão: 0.20)
  --seed S            Semente base do xorshift32 (padrão: 42)
  --dot FILE          Exporta uma árvore em Graphviz DOT
  --dot-tree N        Índice da árvore exportada (padrão: 0)
```

O número de threads vem de `OMP_NUM_THREADS`.
