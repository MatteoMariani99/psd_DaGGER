# Progetto Speciale per la Didattica


## Descrizione
L'obiettivo di questo progetto è la navigazione autonoma di una **auto da modellismo** su un tracciato di strada asfaltata e uno di coni (gialli e blu) per delimitare la corsia.
Il primo passo è stato quello di studiare lo stato dell’arte in modo da reperire quante più informazioni possibili: in particolare viene preso come riferimento il seguente articolo 
(A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning. Stéphane Ross, Geoffrey J. Gordon, J. Andrew Bagnell) in modo da avere una linea guida per lo sviluppo e l’implementazione.


## Struttura
La repository è così strutturata:
- **Materiale**: racchiude gli articoli dello stato dell'arte per lo studio del problema;
- **World&car**: racchiude tutti i file urdf e sdf per il caricamento dell'auto e del tracciato;
- **Scripts**: TO DO;
- **data_test**: contiene tutte le imamgini ottenute (dataset aggregato);
- **dagger_test_models**: contiene i modelli allenati durante la procedura di training;
- **world**: contiene il tracciato dei coni.



## Implementazione
L’articolo scelto presenta descrive una policy denominata Dagger (Dataset Aggregation) un algoritmo iterativo che allena una policy deterministica basandosi sulle osservazioni ottenute dalla guida di un esperto (umano, p, mpc...).
Inizialmente si crea un dataset collezionando immagini dall'ambiente (sotto la policy del solo expert). Dopo aver collezionato N immagini, si procede con un primo training della policy per aggiornare i pesi della rete e minimizzare la loss tramite MSE (azioni expert - azioni predette).
Una volta allenata la prima policy, le azioni intraprese sull'ambiente non saranno date solo dall'esperto ma in parte anche dalla rete, spiegazione:

policy = a * beta + predette * (1-beta)

dove a rappresenta le azioni dell'expert e predette quelle della rete; il coefficiente beta rappresenta una sorta di peso (inizialmente beta = 1 e quindi abbiamo solo azioni dell'expert: finito il primo training beta = 0.9 e così via). 
Se si vogliono eseguire un numero elevato di iterazioni, alla fine beta sarà pari a 0 e ci saranno solo le azioni della rete a comandare l'auto.

![Immagine dagger](https://github.com/MatteoMariani99/psd_DaGGER/blob/main/materiale/dagger.png)


