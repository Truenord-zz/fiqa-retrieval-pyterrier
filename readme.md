Retrieval experiments on FiQA (BEIR) with PyTerrier: indexing and ranking models, query expansion, query expansion with flan-t5 and re-ranking with a bi-encoder.

To run the code, first set up the environment:

    conda env create -f environment.yml && conda activate searchengines

The environment includes Java 11. If PyTerrier cannot find Java, set JAVA_HOME to a Java 11 installation.

The data is downloaded automatically by ir_datasets the first time a notebook runs. Start Jupyter from this folder and run the notebooks in order. Notebooks 2 to 4 read the indexes and the tuned settings that notebook 1 writes to indexes/.

1) 01_indexing_and_ranking.ipynb
2) 02_query_expansion.ipynb
3) 03_llm_query_expansion.ipynb
4) 04_reranking.ipynb

The result tables are saved as csv files in results/. helpers.py has the data loading and the retriever setup shared by the notebooks.
