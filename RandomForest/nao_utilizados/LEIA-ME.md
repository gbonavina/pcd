# Material não utilizado no artigo final

Estes arquivos fizeram parte do desenvolvimento, mas não aparecem no artigo
(`../report/template/artigo.pdf`). Os caminhos relativos foram preservados:
`nao_utilizados/cuda/` estava em `cuda/`, `nao_utilizados/data/` em `data/`, e
assim por diante.

| Caminho | O que é | Por que ficou de fora |
| :------ | :------ | :-------------------- |
| `src/`, `Makefile`, `rf.exe` | Versão sequencial em C11 e o Makefile antigo da raiz | O artigo usa como linha de base o próprio binário OpenMP com 1 thread |
| `cuda/` | Versão CUDA (treino e inferência na GPU) | O artigo trata só de OpenMP |
| `README.md` | README antigo, com a comparação Seq/OpenMP/CUDA | Descreve as três versões e resultados que não estão no artigo |
| `BENCHMARK_RESULTS.md`, `benchmark_results.csv` | Benchmarks comparativos Seq/OpenMP/CUDA | Idem |
| `ENERGY_MEASUREMENT.md` | Metodologia de energia para CPU e GPU (RAPL e NVML) | A metodologia usada está descrita no próprio artigo |
| `python/benchmark_comparison.py` | Script que gerou os benchmarks comparativos | Idem |
| `python/results_comparison.ipynb` | Notebook de comparação com o scikit-learn | Não entrou no artigo |
| `data/sales_data.csv` | Base sintética de vendas | Não é uma das quatro bases do artigo |
| `data/letter+recognition/` | Download bruto do UCI | `letter-recognition.data` é idêntico a `../data/letter-recognition.data` |
| `tree0.dot`, `tree0.png`, `openmp/*.dot`, `openmp/tree0.png` | Saídas avulsas de `--dot` | A Figura 1 usa `report/template/figuras/arvore_iris.*` |
| `report/README.md`, `report/template/sbc-template.*`, `fig*.jpg`, `table.jpg` | Exemplo e instruções do template SBC | Só o estilo (`sbc-template.sty`, `sbc.bst`) é usado |
| `report/template/final_page-*.png`, `page_view-*.png` | Capturas de páginas de versões anteriores do PDF | Não são figuras do artigo |

O código aqui não é mantido: `python/benchmark_comparison.py` e o `Makefile` ainda
apontam para os caminhos antigos.
