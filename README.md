# RAG Against the Machine

RAG simple pour le repo vLLM : indexation des fichiers, recherche BM25,
evaluation recall@k et generation de reponses avec `Qwen/Qwen3-0.6B`.

## Installation

```bash
uv sync
mkdir -p data/raw && unzip -q -n info/given/vllm-0.10.1.zip -d data/raw
```

## Architecture

Le pipeline est separe par famille de commande :

- `src/student/index/` : lecture des fichiers, chunking, index BM25
- `src/student/search/` : chargement de l'index et retrieval top-k
- `src/student/evaluate/` : calcul recall@k par overlap de sources
- `src/student/answer/` : construction du contexte et appel Qwen
- `src/student/app/` : commandes Fire exposees par famille

Flux principal :

```text
data/raw/vllm-0.10.1 -> index -> data/processed
question -> search -> retrieved_sources
retrieved_sources -> answer -> Qwen/Qwen3-0.6B -> answer
retrieved_sources + dataset answered -> evaluate -> recall@k
```

## Chunking

La taille maximale est configurable avec `MAX_CHUNK_SIZE`, bornee a `2000`.

Strategies :

- Python : parse AST et chunk des classes/fonctions top-level. Si le fichier
  n'est pas parsable, fallback en chunks texte.
- Markdown : split par sections `#`, `##`, `###`, puis fallback texte si une
  section est trop grande.
- Texte : split fixe par caracteres pour les autres fichiers lisibles.

Des chunks trop petits perdent le contexte et multiplient le bruit. Des chunks
trop gros diminuent la precision du retrieval et coutent plus cher au LLM.

## Retrieval

Le retrieval utilise BM25 via `bm25s`.

Tokenisation maison :

- lowercase
- split `snake_case`
- split `CamelCase`
- conservation des identifiants et chiffres utiles

Au moment de l'indexation, BM25 indexe :

```text
parent_directory + filename + filename + chunk_text
```

Le nom de fichier est pondere car beaucoup de questions mentionnent des modules,
fichiers, endpoints ou concepts proches du chemin.

## Answer

Le modele par defaut est :

```text
Qwen/Qwen3-0.6B
```

Le modele est charge une seule fois par commande `answer` ou `answer_dataset`.
Par defaut, `answer` recupere 3 chunks pour rester rapide en usage manuel. Si
la commande recoit `--k 10`, elle recupere bien 10 chunks. Le contexte envoye
au modele garde les chunks entiers, avec une limite interne simple :

```text
10 sources max
2000 caracteres max par source
```

Cela respecte la taille maximale des chunks du sujet et evite de couper les
reponses utiles situees en fin de source. Le prompt interdit les liens Markdown,
mais conserve les URLs/endpoints quand la question les demande.

## Performances

Resultats prives avec le script `exam_retrieval.sh` :

```text
Indexing: 5s / limite 300s
Search 200 questions: 1s / limite 90s
Docs recall@5: 0.81 / seuil 0.80
Code recall@5: 0.72 / seuil 0.50
```

Resultats publics observes :

```text
Docs recall@5: ~0.83
Code recall@5: ~0.78
```

## Choix et trade-offs

BM25 a ete choisi plutot qu'un systeme embeddings pour rester simple, rapide et
robuste sans GPU. Le principal trade-off est que BM25 depend fortement des
tokens presents dans la question et dans les chunks. Le tokenizer et la
ponderation du nom de fichier compensent ce probleme pour le code.

Le chunking Python AST evite de couper au milieu des fonctions/classes, mais il
ignore certains morceaux top-level. Ce choix reste volontaire pour garder un
code simple et de bonnes performances.

## Commandes principales

```bash
make index
make search QUERY="How to configure OpenAI server?" K=10
make search_dataset DATASET_PATH=data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json K=10
make evaluate STUDENT_RESULTS_PATH=data/output/search_results/dataset_docs_public.json DATASET_PATH=data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json K=10
make answer QUERY="What is vLLM?" ANSWER_K=3 MAX_NEW_TOKENS=64
make answer_dataset STUDENT_SEARCH_RESULTS_PATH=data/output/search_results/dataset_docs_public.json MAX_NEW_TOKENS=64
```

## Test recall complet

```bash
make index OUTPUT_DIR=/tmp/rag-recall
make search_dataset DATASET_PATH=data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json PROCESSED_DIR=/tmp/rag-recall SAVE_DIRECTORY=/tmp/rag-recall-results K=10
make evaluate STUDENT_RESULTS_PATH=/tmp/rag-recall-results/dataset_docs_public.json DATASET_PATH=data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json K=10
make search_dataset DATASET_PATH=data/datasets_public/public/AnsweredQuestions/dataset_code_public.json PROCESSED_DIR=/tmp/rag-recall SAVE_DIRECTORY=/tmp/rag-recall-results K=10
make evaluate STUDENT_RESULTS_PATH=/tmp/rag-recall-results/dataset_code_public.json DATASET_PATH=data/datasets_public/public/AnsweredQuestions/dataset_code_public.json K=10
```
