import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_dataset(dataset_name):
    print(f"Evaluating dataset {dataset_name}")
    d = np.load('datasets/'+dataset_name+'.npz')
    questions, contexts, most_relevant = d["questions"], d["contexts"], d["most_relevant_context"]

    embeddings = ["gemini_3072", "all-mpnet-base-v2", "multi-qa-MiniLM-L6-dot-v1"]
    for embedding in embeddings:
        question_embeddings = np.load(f"embeddings/{dataset_name}_questions_{embedding}.npz")["embeddings"]
        context_embeddings = np.load(f"embeddings/{dataset_name}_contexts_{embedding}.npz")["embeddings"]
        top1_correct = 0
        reciprocal_ranks = []
        similarities = cosine_similarity(question_embeddings, context_embeddings)
        for i in range(len(questions)):
            sim_scores = similarities[i]
            ranked_indices = np.argsort(-sim_scores)  # descending

            gold_index = most_relevant[i]

            # Top-1 Accuracy
            if ranked_indices[0] == gold_index:
                top1_correct += 1

            # Mean Reciprocal Rank (MRR)
            rank = np.where(ranked_indices == gold_index)[0][0] + 1
            reciprocal_ranks.append(1.0 / rank)

        top1_accuracy = top1_correct / len(questions)
        mrr = np.mean(reciprocal_ranks)

        print(f"Method: {embedding}")
        print(f"Samples: {len(questions)}")
        print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
        print(f"MRR: {mrr:.4f}")

def main():
    evaluate_dataset('squad_1_1')
    evaluate_dataset('assistive_technology_320')

if __name__ == '__main__':
    main()