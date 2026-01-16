import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import os
from csv import reader, writer


def assistive_tech():
    model = load_model()
    with open('assistive_technology_notes_truly_unique.csv') as f:
        facts = reader(f)
        all_facts = [fact[1] for fact in facts]
        fact_embeddings = model.encode(
            all_facts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )
    print(fact_embeddings)

    with open('assistive_tech_topics_with_top3_notes.csv') as f:
        topics = reader(f)
        parsed = [(topic[0], topic[1]) for topic in topics]
        pseudo_questions, top_related_facts = zip(*parsed)
        pseudo_questions_embeddings = model.encode(
            pseudo_questions,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )
    similarities = cosine_similarity(pseudo_questions_embeddings, fact_embeddings)
    with open('assistive_tech_results.csv', 'w') as csvfile:
        results_writer = writer(csvfile)
        results_writer.writerow(['topic','ground truth', 'most similar'])
        for i in range(len(pseudo_questions_embeddings)):
            sim_scores = similarities[i]
            ranked_indices = np.argsort(-sim_scores)
            results_writer.writerow([pseudo_questions[i], top_related_facts[i], all_facts[ranked_indices[0]]])

def load_model():
    model = SentenceTransformer("all-mpnet-base-v2")
    return model


def main():
    print("Loading model...")
    model = load_model()
    # -------------------------------
    # Load SQuAD 1.1
    # -------------------------------
    dataset = load_dataset("squad", split='validation')

    # -------------------------------
    # Prepare data
    # -------------------------------
    questions = dataset["question"]
    contexts = dataset["context"]

    # Build unique context set (many questions share the same paragraph)
    unique_contexts = list(dict.fromkeys(contexts))

    print(f"Questions: {len(questions)}")
    print(f"Unique contexts: {len(unique_contexts)}")
    if os.path.exists("embeddings.npz"):
        data = np.load("embeddings.npz")
        context_embeddings = data["context_embeddings"]
        question_embeddings = data["question_embeddings"]
    else:
        # -------------------------------
        # Encode
        # -------------------------------
        print("Encoding contexts...")
        context_embeddings = model.encode(
            unique_contexts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        print("Encoding questions...")
        question_embeddings = model.encode(
            questions,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        np.savez("embeddings.npz", question_embeddings=question_embeddings, context_embeddings=context_embeddings)

    # Map each question to its correct context index
    context_to_index = {c: i for i, c in enumerate(unique_contexts)}
    gold_context_indices = [context_to_index[c] for c in contexts]

    # -------------------------------
    # Similarity Search
    # -------------------------------
    print("Computing cosine similarities...")
    similarities = cosine_similarity(question_embeddings, context_embeddings)

    # -------------------------------
    # Metrics
    # -------------------------------
    top1_correct = 0
    reciprocal_ranks = []

    for i in range(len(questions)):
        sim_scores = similarities[i]
        ranked_indices = np.argsort(-sim_scores)  # descending

        gold_index = gold_context_indices[i]

        # Top-1 Accuracy
        if ranked_indices[0] == gold_index:
            top1_correct += 1

        # Mean Reciprocal Rank (MRR)
        rank = np.where(ranked_indices == gold_index)[0][0] + 1
        reciprocal_ranks.append(1.0 / rank)

    top1_accuracy = top1_correct / len(questions)
    mrr = np.mean(reciprocal_ranks)

    # -------------------------------
    # Results
    # -------------------------------
    print("\n=== SQuAD 1.1 Retrieval Evaluation ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Split: {SPLIT}")
    print(f"Samples: {len(questions)}")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"MRR: {mrr:.4f}")

if __name__ == "__main__":
    #main()
    assistive_tech()
