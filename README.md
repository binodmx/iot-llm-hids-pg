# LLM-driven Policy Generation for IoT Intrusion Detection Systems

## Introduction

In this research we try to find how Large Language Models (LLMs) perform to improve the IoT security and access control. In particular, we investigate whether LLMs can generate effective policies (rules) to detect intrusion attacks in IoT systems. This experiment integrates Retrieval-Augmented Generation (RAG) and function calling capabilities of LLMs to build a novel policy generation framework for IoT Intrusion Detection Systems.

## Getting Started

1. Clone the repository.
2. Install required Python libraries using `pip install -r requirements.txt`.
3. Create `.env` file in the root directory and add API keys as follows.

    ```bash
    OPENAI_API_KEY=
    GOOGLE_API_KEY=
    ANTHROPIC_API_KEY=
    LANGCHAIN_API_KEY=
    LANGCHAIN_PROJECT=
    LANGCHAIN_TRACING_V2=
    ```

4. Create `data` directory in the root directory and sub directories for datasets as follows.

    ```
    data
    |-cic-iot
    |-wustl-iiot
    |-ton-iot
    |-bot-iot
    └-unsw-nb15
    ```

5. Place downloaded datasets in the relevant sub directories.

6. Run Python notebooks in order for each dataset.

    > For ex:
    >
    > Run `00-dataset-analysis.ipynb`, `01-preprocessing.ipynb`, `02-baseline-ml.ipynb`, `03-multiclass-classification-llm.ipynb`, `04-zero-day-detection-claude.ipynb`, `05-anonymization-ablation-original.ipynb`, `06-binary-classification-llm-claude.ipynb`, ... in `1-cic-iot` directory to evaluate `CICIoT2023` dataset. Each dataset folder follows the same numbering, matching the order results appear in Section V of the paper; superseded/exploratory notebooks live under each folder's `archive/` subdirectory.

## Datasets

| Name | Paper(s) | Year |
| - | - | - |
| CICIoT2023* | CICIoT2023: A Real-Time Dataset and Benchmark for Large-Scale Attacks in IoT Environment | 2023 |
| Edge-IIoTSet | Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications for Centralized and Federated Learning | 2022 |
| WUSTL-IIoT* | WUSTL-IIOT-2021 Dataset for IIoT Cybersecurity Research | 2021 |
| IoT-23 | IoT-23: A labeled dataset with malicious and benign IoT network traffic | 2020 |
| TON_IoT* | TON_IoT telemetry dataset: a new generation dataset of IoT and IIoT for data-driven Intrusion Detection Systems | 2020 |
| Bot-IoT* | Towards the development of realistic botnet dataset in the internet of things for network forensic analytics: Bot-iot dataset | 2019 |
| N-BaIoT | N-BaIoT: Network-based Detection of IoT Botnet Attacks Using Deep Autoencoders | 2018 |
| UNSW-NB15* | UNSW-NB15: A Comprehensive Data set for Network Intrusion Detection systems | 2015 |

## Large Language Models

| Name               | Provider  |
|--------------------|-----------|
| gpt-4o*            | OpenAI    |
| gemini-1.5-pro*    | Google    |
| claude-3-5-sonnet* | Anthropic |

\* Used in our experiment.