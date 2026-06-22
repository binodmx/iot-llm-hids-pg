# Multi-Seed Policy Pipeline Results (3 seeds: 42, 123, 456)

Mean ± std on the held-out test split, per dataset × LLM vendor.

## Macro F1

| Dataset | Claude Haiku 4.5 | Gemini 2.5 Flash |
|---|---|---|
| CIC-IoT2023 | 0.9780 ± 0.0050 | 0.9823 ± 0.0059 |
| WUSTL-IIoT | 0.9803 ± 0.0070 | 0.9854 ± 0.0039 |
| TON\_IoT | 0.7696 ± 0.0599 | 0.8494 ± 0.0306 |
| Bot-IoT | 0.9820 ± 0.0170 | 0.9707 ± 0.0195 |
| UNSW-NB15 | 0.6948 ± 0.0140 | 0.7040 ± 0.0021 |

## Attack-Class F1

| Dataset | Haiku | Gemini |
|---|---|---|
| CIC-IoT2023 | 0.9780 ± 0.0049 | 0.9821 ± 0.0060 |
| WUSTL-IIoT | 0.9803 ± 0.0072 | 0.9855 ± 0.0039 |
| TON\_IoT | 0.8093 ± 0.0434 | 0.8546 ± 0.0175 |
| Bot-IoT | 0.9823 ± 0.0167 | 0.9700 ± 0.0202 |
| UNSW-NB15 | 0.7685 ± 0.0276 | 0.7727 ± 0.0223 |

## Attack-Class Precision / Recall (Haiku)

| Dataset | Precision | Recall |
|---|---|---|
| CIC-IoT2023 | 0.9817 ± 0.0160 | 0.9745 ± 0.0106 |
| WUSTL-IIoT | 0.9781 ± 0.0027 | 0.9826 ± 0.0171 |
| TON\_IoT | 0.7093 ± 0.0541 | 0.9432 ± 0.0229 |
| Bot-IoT | 0.9738 ± 0.0260 | 0.9910 ± 0.0078 |
| UNSW-NB15 | 0.6433 ± 0.0040 | 0.9562 ± 0.0758 |

## Attack-Class Precision / Recall (Gemini)

| Dataset | Precision | Recall |
|---|---|---|
| CIC-IoT2023 | 0.9878 ± 0.0013 | 0.9766 ± 0.0111 |
| WUSTL-IIoT | 0.9744 ± 0.0027 | 0.9969 ± 0.0051 |
| TON\_IoT | 0.8444 ± 0.0880 | 0.8733 ± 0.0553 |
| Bot-IoT | 0.9863 ± 0.0139 | 0.9550 ± 0.0413 |
| UNSW-NB15 | 0.6513 ± 0.0077 | 0.9533 ± 0.0810 |
