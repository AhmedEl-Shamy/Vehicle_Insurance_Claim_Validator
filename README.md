# Vehicle Insurance Claim Validator

An AI-powered multimodal system for **vehicle insurance claim validation and explainable fraud-risk assessment**.

The system analyzes different types of claim evidence, including **vehicle damage images, user claim descriptions, police reports, and repair invoices/estimates**. Each type of evidence is processed by a dedicated AI module, and the extracted information is later compared to identify inconsistencies and potential risk indicators.

## Project Overview

Insurance claims usually contain information from multiple sources. Reviewing these sources manually can be time-consuming and may make it difficult to identify contradictions between them.

This project aims to provide a unified system that automatically processes the available evidence, converts it into structured information, and performs cross-modal consistency analysis to support insurance investigators.

The system is designed as a **decision-support tool** and does not replace human investigators.

## Main Components

### 1. Vehicle Image Processing

The Vision Module analyzes submitted vehicle images to identify visible damage and the affected vehicle parts.

The pipeline includes:

- Image quality validation
- Vehicle damage detection
- Vehicle-part segmentation
- Damage-to-part matching
- Damage severity estimation

The module uses trained **YOLOv11-based models** and produces structured JSON/CSV outputs containing information such as damage type, confidence, bounding box, affected part, and severity.

### 2. User Claim Description

The Claim Description Module processes free-text accident descriptions and extracts structured information such as:

- Vehicle make and model
- Model year
- Accident date and time
- Damage description
- Damaged parts

**Model:** Qwen2.5-1.5B-Instruct

The extracted information is generated as JSON and validated using a **Pydantic schema** to ensure that the output follows the required structure and data types.

### 3. Police Report Processing

The Police Report Module processes police accident reports and extracts structured information from their visual layouts.

**Model:** Qwen2.5-VL-7B-Instruct

PDF pages are converted into images so that the model can preserve the spatial structure of the original reports.

The module extracts information such as:

- Accident details
- Driver information
- Vehicle information
- Damaged parts
- General damage areas

### 4. Invoice and Repair Estimate Processing

This module processes repair estimates and body shop invoices to extract structured financial and component-level information.

**Model:** Gemini 3.6 Flash

The module uses high-resolution PDF rendering, spatial prompting, Pydantic schema validation, taxonomy normalization, and JSON-repair mechanisms to produce structured outputs.

### 5. Cross-Modal Consistency and Reasoning

The outputs from the four processing modules are compared to identify:

- Agreements between evidence sources
- Contradictions
- Missing evidence
- Potential risk indicators

The system evaluates consistency across the available evidence and produces a **Low, Medium, or High fraud-risk level** together with an explainable report.

## System Workflow

```text
                    ┌─────────────────────┐
                    │   Vehicle Images    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Vision Module     │
                    └──────────┬──────────┘
                               │
                               │
┌──────────────────┐  ┌────────▼──────────┐  ┌──────────────────────┐
│ Claim Description│  │ Structured        │  │ Police Reports      │
│                  │  │ Evidence          │  │ + Invoices          │
└────────┬─────────┘  └────────┬───────────┘  └──────────┬───────────┘
         │                     │                         │
         ▼                     │                         ▼
┌──────────────────┐            │              ┌──────────────────────┐
│ Claim NLP Module │            │              │ Document Modules     │
└────────┬─────────┘            │              └──────────┬───────────┘
         │                      │                         │
         └──────────────────────┼─────────────────────────┘
                                ▼
                  ┌─────────────────────────┐
                  │ Cross-Modal Consistency │
                  │      & Reasoning        │
                  └────────────┬────────────┘
                               │
                               ▼
                  ┌─────────────────────────┐
                  │  Risk Assessment &      │
                  │  Explainable Report     │
                  └─────────────────────────┘