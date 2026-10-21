# Vehicle Insurance Claim Validator

A multimodal AI system for **vehicle insurance claim validation and explainable fraud-risk assessment**.

The system reads four types of claim evidence: **vehicle damage images, user claim descriptions, repair invoices, and police reports**. Each type goes through its own AI pipeline. The outputs are compared across sources to find matches, missing details, and contradictions. Each claim gets a **Low, Medium, or High** risk level and an explainable report.

The system is a **decision-support tool**. It does not replace human investigators.

## Team

- Natalie Monged Maurice
- Mariam Anwer Mohammed
- Ahmed Alshamy
- Bassel Essa

**Supervisor:** Prof. Hazem Abbas, School of Computing, Queen's University

## Project Idea

Insurance claims have evidence from many sources. Checking them by hand is slow and can miss errors. Most existing AI tools use one data type, or they act as black boxes with no clear reasoning. Also, generative AI can now create fake damage photos.

This project solves these problems with:

- An input validation layer, including AI-generated image detection
- Four specialized extraction pipelines
- Pairwise cross-modal consistency checks
- A deterministic risk engine
- An evidence-based report for the human reviewer
- A web app with a user portal and an admin dashboard

## System Layers

### 1. Input Validation Layer

Runs before any expensive AI step.

- Image checks: resolution, blur, brightness, contrast, perceptual quality, and vehicle presence
- Synthetic image detection (Stable Diffusion traces)
- Claim completeness and document readability checks
- Invalid inputs are rejected or flagged for review

### 2. Input Analysis Layer (4 pipelines)

| Pipeline | Output |
|---|---|
| Vehicle images | Damage type, confidence, box, affected part, severity |
| Claim description | Car info and damaged parts as validated JSON |
| Invoice / repair estimate | Car info and repaired parts with operations |
| Police report | Accident details, driver and vehicle info, damaged parts |

### 3. Cross-Modal Consistency Layer

The four JSON outputs are compared in **six pairs**: image–invoice, image–description, image–police, description–invoice, description–police, invoice–police.

For each pair:

1. Preprocess each source into a clean list of part names
2. An LLM matches the parts of the two lists
3. The answer is validated and retried if needed
4. Code finds missing parts and checks car make, model, and year
5. Code calculates a pairwise score

The overall consistency score is the mean of the six pairwise scores.

### 4. Risk Analysis Layer

A deterministic decision engine (TypeSafe JEV) uses the consistency results and the image signals (AI-image threat and vehicle detection).

| Risk | Conditions |
|---|---|
| **High** | High AI-image threat, visible ghost billing, consistency score < 0.60, or no vehicle detected |
| **Medium** | Mild AI threat, low vehicle confidence, or consistency score 0.60 – 0.79 |
| **Low** | No AI threat, vehicle confirmed, no ghost billing, and consistency score ≥ 0.80 |

If the engine fails, the claim defaults to **Medium** and goes to manual human review.

### 5. Explainable Report Layer

An LLM writes a Markdown report with five sections: executive summary, key findings, risk assessment, recommended next steps, and overall recommendation. The report keeps the risk level from the engine and separates risk indicators from confirmed fraud.

### 6. Human Review

The investigator dashboard shows images, masks, structured claims, risk score, and the explanation. The investigator can approve, reject, or send the claim for further investigation.

## System Workflow

```text
   Vehicle Images        Claim Description       Invoice (PDF)      Police Report (PDF)
         │                      │                      │                    │
         └──────────────────────┴──────────┬───────────┴────────────────────┘
                                           ▼
                            ┌─────────────────────────────┐
                            │   Input Validation Layer    │
                            │ quality · AI-image · format │
                            └──────────────┬──────────────┘
                                           ▼
        ┌──────────────┬──────────────────┬─────────────────┬──────────────────┐
        │ Image        │ Text             │ Invoice         │ Police Report    │
        │ YOLOv11      │ Qwen2.5-1.5B     │ Gemini 3.6      │ Qwen2.5-VL-7B    │
        └──────┬───────┴────────┬─────────┴────────┬────────┴─────────┬────────┘
               └────────────────┴────────┬─────────┴──────────────────┘
                                         ▼
                        ┌──────────────────────────────┐
                        │ Consistency Layer (6 pairs)  │
                        └──────────────┬───────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │ Risk Analysis (TypeSafe JEV) │
                        └──────────────┬───────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │ Explainable Report (Gemini)  │
                        └──────────────┬───────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │ Human Review (Approve /      │
                        │ Investigate / Reject)        │
                        └──────────────────────────────┘
```

## Models Used

| Layer | Task | Model |
|---|---|---|
| Validation | Image quality | MANIQA |
| Validation | AI-generated image detection | Mandelli et al. detector (PoliMi) |
| Validation / Vision | Vehicle presence | YOLOv11 |
| Image pipeline | Damage detection | YOLOv11 (fine-tuned on CarDD) |
| Image pipeline | Vehicle-part segmentation | YOLO11n-seg |
| Claim description | Information extraction | Qwen2.5-1.5B-Instruct |
| Police report | Document understanding | Qwen2.5-VL-7B-Instruct (4-bit) |
| Invoice | Document understanding | Gemini 3.6 Flash |
| Invoice | Fallback / validation | GPT-6 Luna (via OpenRouter) |
| Consistency | Part matching | openai/gpt-oss-20b (via Groq) |
| Risk analysis | Risk classification | TypeSafe JEV (`typesafe/jev-latest`) |
| Report | Report generation | Gemini |

## How to Run

1. Install the packages:

```bash
pip install -r requirements.txt
```

2. Download and save the models by running the script:

```bash
python save_models.py
```

3. Start the server:

```bash
uvicorn main:app --reload
```