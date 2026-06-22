make index OUTPUT_DIR=/tmp/rag-recall
make search_dataset DATASET_PATH=data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json PROCESSED_DIR=/tmp/rag-recall SAVE_DIRECTORY=/tmp/rag-recall-results K=10
make evaluate STUDENT_RESULTS_PATH=/tmp/rag-recall-results/dataset_docs_public.json DATASET_PATH=data/datasets_public/public/AnsweredQuestions/dataset_docs_public.json K=10
make search_dataset DATASET_PATH=data/datasets_public/public/AnsweredQuestions/dataset_code_public.json PROCESSED_DIR=/tmp/rag-recall SAVE_DIRECTORY=/tmp/rag-recall-results K=10
make evaluate STUDENT_RESULTS_PATH=/tmp/rag-recall-results/dataset_code_public.json DATASET_PATH=data/datasets_public/public/AnsweredQuestions/dataset_code_public.json K=10
