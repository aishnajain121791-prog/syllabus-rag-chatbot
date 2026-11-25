# Evaluation Results

- **Answerable Questions**: 25
- **Trap (Unanswerable) Questions**: 5
- **Similarity Threshold (`MIN_SCORE`)**: `0.30`

## Summary Metrics

| Metric | Score |
| --- | --- |
| Hit@1 | 72.0% (18/25) |
| Hit@3 | 84.0% (21/25) |
| MRR | 0.7733 |
| Expected-Chunk Accuracy | 33.3% (25/75) |
| Trap Refusal Rate | 20.0% (1/5: 0 threshold, 1 LLM) |
| Trap Non-Refusals / API Errors | 0 answered / 4 API errors (answerable API errors: 5) |

## Answerable Questions Breakdown

| ID | Expected Source & Page | Hit@1 | Hit@3 | RR | Matching Chunks |
| --- | --- | :---: | :---: | ---: | ---: |
| Q1 | COMPUTER ORGANIZATION (R17A0510).pdf (p.2) | YES | YES | 1.00 | 1/3 |
| Q2 | COMPUTER ORGANIZATION (R17A0510).pdf (p.15) | YES | YES | 1.00 | 2/3 |
| Q3 | COMPUTER ORGANIZATION (R17A0510).pdf (p.88) | NO | YES | 0.50 | 1/3 |
| Q4 | COMPUTER ORGANIZATION AND ARCHITECTURE.pdf (p.1) | YES | YES | 1.00 | 1/3 |
| Q5 | COMPUTER ORGANIZATION AND ARCHITECTURE.pdf (p.35) | YES | YES | 1.00 | 1/3 |
| Q6 | COMPUTER ORGANIZATION AND ARCHITECTURE.pdf (p.58) | YES | YES | 1.00 | 2/3 |
| Q7 | COMPUTER ORGANIZATION AND ARCHITECTURE.pdf (p.70) | YES | YES | 1.00 | 1/3 |
| Q8 | COMPUTER ORGANIZATION NOTES.pdf (p.4) | YES | YES | 1.00 | 1/3 |
| Q9 | COMPUTER ORGANIZATION NOTES.pdf (p.47) | YES | YES | 1.00 | 1/3 |
| Q10 | COMPUTER ORGANIZATION NOTES.pdf (p.71) | YES | YES | 1.00 | 1/3 |
| Q11 | COMPUTER ORGANIZATION (R17A0510).pdf (p.25) | YES | YES | 1.00 | 1/3 |
| Q12 | COMPUTER ORGANIZATION (R17A0510).pdf (p.42) | YES | YES | 1.00 | 1/3 |
| Q13 | COMPUTER ORGANIZATION (R17A0510).pdf (p.53) | NO | NO | 0.00 | 0/3 |
| Q14 | COMPUTER ORGANIZATION (R17A0510).pdf (p.62) | YES | YES | 1.00 | 1/3 |
| Q15 | COMPUTER ORGANIZATION (R17A0510).pdf (p.75) | NO | NO | 0.00 | 0/3 |
| Q16 | COMPUTER ORGANIZATION AND ARCHITECTURE.pdf (p.15) | YES | YES | 1.00 | 1/3 |
| Q17 | COMPUTER ORGANIZATION AND ARCHITECTURE.pdf (p.46) | YES | YES | 1.00 | 1/3 |
| Q18 | COMPUTER ORGANIZATION AND ARCHITECTURE.pdf (p.64) | YES | YES | 1.00 | 2/3 |
| Q19 | COMPUTER ORGANIZATION AND ARCHITECTURE.pdf (p.81) | NO | NO | 0.00 | 0/3 |
| Q20 | COMPUTER ORGANIZATION AND ARCHITECTURE.pdf (p.90) | YES | YES | 1.00 | 2/3 |
| Q21 | COMPUTER ORGANIZATION NOTES.pdf (p.13) | YES | YES | 1.00 | 1/3 |
| Q22 | COMPUTER ORGANIZATION NOTES.pdf (p.26) | NO | NO | 0.00 | 0/3 |
| Q23 | COMPUTER ORGANIZATION NOTES.pdf (p.37) | NO | YES | 0.50 | 1/3 |
| Q24 | COMPUTER ORGANIZATION NOTES.pdf (p.53) | YES | YES | 1.00 | 1/3 |
| Q25 | COMPUTER ORGANIZATION NOTES.pdf (p.61) | NO | YES | 0.33 | 1/3 |

## Trap Questions Breakdown

| ID | Question | Classification | Top Score Returned |
| --- | --- | --- | ---: |
| T1 | How does the Adam optimizer update neural network weights using first and second moment estimates? | llm_refusal | 0.315 |
| T2 | What do the four ACID properties guarantee for database transactions? | api_error | 0.387 |
| T3 | How does Python's Global Interpreter Lock (GIL) affect CPU-bound multithreaded programs? | api_error | 0.516 |
| T4 | What is the balanced chemical equation for photosynthesis in plants? | api_error | 0.348 |
| T5 | How does TCP slow start dynamically adjust the congestion window size? | api_error | 0.366 |
