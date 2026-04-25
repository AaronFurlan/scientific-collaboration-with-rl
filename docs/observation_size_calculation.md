# Analyse der Observationsgröße (Feature-Vektor)

Diese Dokumentation erklärt die korrekte Berechnung der **451 Features**, die das Modell in Ihrem Checkpoint erwartet.

## 1. Parameter im Checkpoint
Basierend auf Ihrem WandB-Log und der Code-Analyse sind dies die korrekten Parameter:
*   `max_peer_group_size (PG)` = **40**
*   `max_projects_per_agent (PA)` = **8**
*   `n_projects_per_step` = **1**

---

## 2. Detaillierte Aufschlüsselung (Warum 8 Projekte = 451 Features ergeben)

### A. Observations-Teil (Zweig: `observation`)
| Komponente | Berechnung | Features |
| :--- | :--- | :--- |
| **Peer-Informationen** | $PG \times 5$ (ID, Rep, H-Index, Centroid X/Y) | 200 |
| **Eigene Statistiken** | Alter (1), Belohnung (1), Centroid X/Y (2) | 4 |
| **Offene Projekte** | $1 \times 4$ (Effort, Prestige, Novelty, Time) | 4 |
| **Laufende Projekte** | $8 \times$ Features pro Slot | **192** |
| *Details pro Slot* | *14 Basis + 10 Identität (Index, IndexNorm, One-Hot-8)* | *(24 pro Slot)* |
| **Gesamt Observation** | | **400** |

### B. Action-Masken Teil (Zweig: `action_mask`)
| Komponente | Berechnung | Features |
| :--- | :--- | :--- |
| **Maske: Projektwahl** | $n\_projects\_per\_step + 1$ | 2 |
| **Maske: Kooperation** | $PG$ (Ein Bit pro Peer) | 40 |
| **Maske: Aufwand** | $PA + 1$ ($8 + 1$ für "nichts tun") | 9 |
| **Gesamt Masken** | | **51** |

---

## 3. Endergebnis
*   **Observation Teil:** 400
*   **Masken Teil:** 51
*   **SUMME:** 400 + 51 = **451 Features**

---

## 4. Fazit
Die mathematische Prüfung bestätigt: **8 Projekte führen bei 40 Peers exakt zu 451 Features.** 

Meine vorherige Vermutung, dass 10 Projekte nötig seien, basierte auf einem Rechenfehler in der Identitäts-Logik der Slots. Die 10 Identitäts-Features pro Slot entstehen bei 8 Projekten durch:
*   1x Slot-Index
*   1x Slot-Index-Normiert
*   8x One-Hot-Vektor (da PA=8)
*   **Summe = 10 Identitäts-Features.**

Zusammen mit den 14 Basis-Features ergeben sich die 24 Features pro Slot, die in der Summe mit den Masken exakt die 451 ergeben. Ihr WandB-Log (PA=8) ist also absolut konsistent mit dem Modell.
