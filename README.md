# Progetto Speciale per la Didattica


## Descrizione
L'obiettivo di questo progetto è la navigazione autonoma di una **auto da modellismo** su un tracciato di strada asfaltata e uno di coni (gialli e blu) per delimitare la corsia.
Il primo passo è stato quello di studiare lo stato dell’arte in modo da reperire quante più informazioni possibili: in particolare viene preso come riferimento il seguente articolo 
(A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell) in modo da avere una linea guida per lo sviluppo e l’implementazione.


## Struttura
La repository è così strutturata:
- **Articoli**: racchiudono gli articoli dello stato dell'arte per lo studio del problema;
- **Relazione**: racchiude tutti i file .tex per la scrittura della relazione;
- **Scripts**: racchiude tutti i file .py utilizzati per la simulazione del problema.

Inoltre, sono presenti la relazione in formato .pdf e la presentazione in formato .pptx.


## Implementazione
L’articolo scelto presenta una policy di collision avoidance decentralizzata che permette di mappare direttamente le misure dei sensori in comandi di sterzo e velocità di avanzamento per ciascun robot.
Per fare ciò viene utilizzato un algoritmo di Reinforcerment Learning (RL) con una politica di tipo Proximal Policy Optimization (PPO) che si basa sul metodo del gradiente. L’algoritmo integra inoltre un controllo ibrido, con 3 diverse modalità di funzionamento, che permette di aumentare la robustezza e l’efficacia del sistema (opzionale).

Per la simulazione dell’algoritmo è stato utilizzato PyBullet, un modulo Python per simulazioni in ambito robotico e dell’apprendimento automatico, con particolare attenzione
al trasferimento da simulazione a realtà: il robot utilizzato in simulazione è il TurtleBot 2.

Viene mostrata la struttura e la suddivisione del codice tra le fasi di testing e training:

![Immagine classi](https://github.com/MatteoMariani99/Progetto-robotica/blob/main/Relazione/Images/final.jpg)

## Installazione
Dopo aver scaricato l'intera cartella eseguire il comando:
```bash
pip install -r requirements.txt
```
In questo modo vengono installate tutte le dipendenze necessarie per il funzionamento del codice.

