Analysis

Collab Analysis:

Über 9 Evaluation-Seeds zeigt der Agent einen systematischen Kollaborations-Bias: Er wählt im Mittel nähere und reputationsstärkere Peers. 
Die Effekte sind jedoch moderat, was auf einfache Feature-Nutzung statt komplexer strategischer Partnerwahl hindeutet.

The knowledge space position is a latent variable influencing reward generation, but not directly observable. 
As a result, the agent is unable to exploit this structure and instead learns policies based on observable proxies.

The agent’s behavior is strongly influenced by structural properties of the observation encoding (e.g., slot ordering), leading to shortcut learning. 
While meaningful signals such as project quality and peer attributes are partially utilized, the decision process is 
dominated by environment-induced biases and limited action diversity.

While the agent could theoretically benefit from rejecting low-quality projects in anticipation of better future opportunities, 
this behavior is difficult to learn due to stochasticity, delayed rewards, and the absence of simultaneous alternatives. 
As a result, the learned policy resembles a threshold-based admission strategy rather than a forward-looking optimization of future opportunities.

## Sensitivitätsanalyse
Die Sensitivitätsanalyse in diesem Projekt ist ein Verfahren zur „Black-Box“-Untersuchung der gelernten RL-Policy (PPO). Sie misst, wie stark sich die Entscheidungen des Agenten ändern, wenn einzelne Eingabewerte (Features) minimal verändert werden.
Hier ist die detaillierte Erklärung des Ablaufs:
1. Grundlage: Das neuronale Netz und Logits
Die Policy des Agenten ist ein neuronales Netz, das einen Beobachtungsvektor (451 Werte) erhält und für jeden Aktions-Head (z. B. choose_project) sogenannte Logits (unnormierte Vorhersagewerte) ausgibt. Diese Logits werden mittels mathematischer Funktionen in Wahrscheinlichkeiten umgewandelt:
•
Softmax: Für diskrete Entscheidungen (z. B. Welches Projekt wird gewählt?).
•
Sigmoid: Für binäre Entscheidungen (z. B. Ja/Nein zur Zusammenarbeit mit Peer X).
2. Der Prozess der Störung (Perturbation)
Um die Sensitivität eines bestimmten Features (z. B. novelty) zu berechnen, nutzt das Notebook die Funktion compute_flat_logit_sensitivity:
1.
Original-Durchlauf: Ein unveränderter Beobachtungsvektor wird durch das Modell geschickt. Man erhält die originalen Aktionswahrscheinlichkeiten $P_{orig}$.
2.
Gezielte Störung: Ein spezifisches Feature im Vektor wird um einen winzigen Wert $\epsilon$ (Epsilon, z. B. $0,001$) erhöht. Alle anderen 450 Werte bleiben gleich.
3.
Gestörter Durchlauf: Der veränderte Vektor wird durch das Modell geschickt. Man erhält neue Wahrscheinlichkeiten $P_{pert}$.
3. Berechnung der Sensitivität
Die Sensitivität ist die durchschnittliche absolute Änderung der Wahrscheinlichkeiten, normiert auf die Größe der Störung: $$\text{Sensitivität} = \frac{\text{mean}(|P_{pert} - P_{orig}|)}{\epsilon}$$
Ein hoher Wert bedeutet, dass das neuronale Netz sehr sensibel auf dieses Feature reagiert – es spielt also eine große Rolle für die aktuelle Entscheidung.
4. Datengrundlage: Frische Beobachtungen
Anstatt alte Logs zu verwenden, generiert das Notebook frische Daten:
•
Die Umgebung wird mit festen Seeds (101–110) gestartet.
•
Der Agent „spielt“ mit der gelernten Policy in der Umgebung.
•
Dabei werden ca. 1000 repräsentative Beobachtungen gesammelt, die direkt aus der aktuellen Modell-Logik stammen. Dies stellt sicher, dass die Dimensionen (451 Features) exakt zum Modell passen.
5. Feature-Mapping
Da der Beobachtungsvektor flach ist (nur eine lange Liste von Zahlen), nutzt die Analyse die Methode get_feature_index_map() aus dem RLLibSingleAgentWrapper. Diese Methode fungiert wie ein Wörterbuch, das Namen wie observation.project_opportunities.project_0.novelty den korrekten Index im 451-Elemente-Vektor zuordnet.
6. Aggregation und Interpretation
Die Ergebnisse werden über alle 1000 Beobachtungen gemittelt und für jeden Aktions-Head separat dargestellt.
•
Beispiel: Wenn project_0 prestige eine hohe Sensitivität beim Head choose_project hat, aber eine niedrige bei put_effort, wissen wir, dass der Agent Prestige nutzt, um zu entscheiden, ob er ein Projekt startet, aber nicht, wie viel Arbeit er danach hineinsteckt.
Zusammenfassend lässt sich sagen: Die Analyse „pikst“ den Agenten bei jedem Feature leicht an und schaut, wie sehr er zusammenzuckt (seine Meinung ändert)


## Average Normalized Sensitivity per Feature and Action Head:

| Action Head $\to$   | choose_project | collaborate_with | put_effort |
|---------------------|-|---|---|
| accumulated_rewards | 0.001065 | 0.001059 | 0.000276 |
| age                 | 0.000690 | 0.000553 | 0.000153 |
| project_0 effort    | 0.000905 | 0.000752 | 0.000268 |
| project_0 novelty   | 0.003422 | 0.000654 | 0.000218 |
| project_0 prestige  | 0.003543 | 0.000600 | 0.000156 |
| project_0 time      | 0.000719 | 0.000692 | 0.000190 |
| running_0 effort    | 0.001306 | 0.000674 | 0.000318 |
| running_0 is_active | 0.003006 | 0.000613 | 0.000178 |
| running_0 time_left | 0.000931 | 0.000625 | 0.000210 |

### Key Insights
The agent’s decision to start a project is primarily driven by a combination of project quality signals (novelty, prestige)
and internal capacity constraints (existing workload), indicating a learned admission control strategy rather than complex long-term planning.

Collaboration decisions exhibit significantly lower sensitivity to individual features, suggesting weaker or noisier 
learning signals in the multi-agent interaction component.

Effort allocation is dominated by features of currently running projects, consistent with a short-term exploitation strategy.

