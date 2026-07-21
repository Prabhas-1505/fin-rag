"""
Builds a real golden evaluation dataset by running actual questions through
the live pdf_search_tool + PDF specialist agent against the ingested document,
capturing the real retrieved contexts and real generated answers. Ground-truth
reference answers are hand-written from the source paper.

Usage:
    PYTHONPATH=. python src/evals/build_golden_dataset.py
"""
import json
import warnings

warnings.simplefilter("ignore")
from dotenv import load_dotenv

load_dotenv()

from src.agents.specialists import create_pdf_specialist
from src.agents.tools import get_searcher, get_reranker

QUESTIONS = [
    {
        "question": "What federated learning aggregation algorithm does the paper use?",
        "ground_truth": "The paper uses the Federated Averaging (FedAvg) algorithm to aggregate model weight updates from edge nodes.",
    },
    {
        "question": "What acoustic classification accuracy does the Audio-CNN achieve?",
        "ground_truth": "The Audio-CNN achieves an acoustic classification accuracy of 95.7%.",
    },
    {
        "question": "What segmentation IoU does the U-Net achieve for canopy clearing detection?",
        "ground_truth": "The ResNet-34 based U-Net achieves a visual segmentation Intersection over Union (IoU) of 0.857.",
    },
    {
        "question": "What model architecture is used for satellite image segmentation?",
        "ground_truth": "A ResNet-34 based U-Net architecture is used for pixel-level satellite image segmentation of canopy clearing.",
    },
    {
        "question": "What edge hardware is used to deploy and benchmark the system?",
        "ground_truth": "The system is deployed and benchmarked on Raspberry Pi 4B, an ARM64 edge device.",
    },
    {
        "question": "How much does federated learning reduce data transmission compared to raw audio streaming?",
        "ground_truth": "Federated Learning reduces data transmission overhead by 95.8% compared to streaming raw audio.",
    },
    {
        "question": "What sampling rate is used to record the acoustic audio signals?",
        "ground_truth": "Raw audio signals are recorded at a sampling rate of 22,050 Hz.",
    },
    {
        "question": "How many Mel filters are used in the Mel-scaling step of the DSP pipeline?",
        "ground_truth": "A bank of 64 triangular Mel filters is applied, spaced between 20 Hz and 8000 Hz.",
    },
]


def main():
    pdf_agent = create_pdf_specialist()
    searcher = get_searcher()
    reranker = get_reranker()

    records = []
    for i, item in enumerate(QUESTIONS, 1):
        q = item["question"]
        print(f"[{i}/{len(QUESTIONS)}] {q}")

        # Real retrieved contexts (same path the agent's tool call uses internally)
        import asyncio
        retrieved = asyncio.run(searcher.hybrid_search(q, top_k=20))
        top_chunks = reranker.rerank(q, retrieved, top_k=4)
        contexts = [c.get("payload", {}).get("text", "") for c in top_chunks if c.get("payload", {}).get("text")]

        # Real generated answer from the live agent
        result = pdf_agent.invoke({"input": q})
        answer = result["output"]

        records.append({
            "question": q,
            "ground_truth": item["ground_truth"],
            "contexts": contexts if contexts else ["(no context retrieved)"],
            "answer": answer,
        })
        print(f"    -> answer: {answer}\n")

    with open("golden_dataset.json", "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nSaved {len(records)} records to golden_dataset.json")


if __name__ == "__main__":
    main()
