"""
Downloads a ~10-document, 500+ page public-domain corpus into data/eval_corpus/
(gitignored - built on demand, not committed) for the SmartRAG benchmark in
src/evals/benchmark_smartrag.py. Sources are NIST Special Publications and
arXiv papers, chosen for free direct-PDF access and no auth/rate limiting.
"""
import os
import urllib.request

OUTPUT_DIR = "data/eval_corpus"

SOURCES = {
    "nist_800-63b_digital_identity.pdf": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-63b.pdf",
    "nist_800-171r2_protecting_cui.pdf": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-171r2.pdf",
    "nist_800-53r5_security_controls.pdf": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf",
    "nist_800-37r2_risk_management.pdf": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-37r2.pdf",
    "nist_800-61r2_incident_handling.pdf": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-61r2.pdf",
    "nist_800-207_zero_trust.pdf": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-207.pdf",
    "arxiv_2312.10997_rag_survey.pdf": "https://arxiv.org/pdf/2312.10997",
    "arxiv_2302.12173_indirect_prompt_injection.pdf": "https://arxiv.org/pdf/2302.12173",
    "arxiv_2306.05499_llm_security_survey.pdf": "https://arxiv.org/pdf/2306.05499",
}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename, url in SOURCES.items():
        dest = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(dest):
            print(f"[skip] {filename} already present")
            continue
        print(f"[fetch] {filename} <- {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as response, open(dest, "wb") as f:
            f.write(response.read())
    print("Done.")


if __name__ == "__main__":
    main()
