# Progetto Speciale per la Didattica


## Descrizione
L'obiettivo di questo progetto è la navigazione autonoma di una **auto da modellismo** su un tracciato di strada asfaltata e uno di coni (gialli e blu) per delimitare la corsia.
Il primo passo è stato quello di studiare lo stato dell’arte in modo da reperire quante più informazioni possibili: in particolare viene preso come riferimento il seguente articolo [Dagger article](https://arxiv.org/pdf/1011.0686) in modo da avere una linea guida per lo sviluppo e l’implementazione.


## Struttura
La repository è così strutturata:
- **docs**: racchiude gli articoli dello stato dell'arte per lo studio del problema;
- **world**: racchiude tutti i file urdf e sdf per il caricamento dell'auto e del tracciato;
- **scripts**: racchiude tutti i file utili al funzionamento dell'algoritmo;
- **data_test**: contiene tutte le imamgini/label ottenute (dataset aggregato);
- **dagger_models**: contiene i modelli allenati durante la procedura di training;
- **runs**: contiene i dati del training riproducibili su TensorBoard;
- **results**: contiene immagini relative al training (loss) e i video di test;
- **trajectory**: contiene tutti i dati delle traiettorie (posizione veicolo) ottenute durante i test;


## Implementazione
L’articolo scelto descrive un algoritmo iterativo denominato Dagger (Dataset Aggregation) che allena una policy deterministica basandosi sulle osservazioni ottenute dalla guida di un esperto (umano, controllore P, mpc...).
Inizialmente si crea un dataset collezionando immagini dall'ambiente (sotto la policy del solo expert). Dopo aver collezionato N immagini, si procede con un primo training della policy per aggiornare i pesi della rete e minimizzare la loss tramite MSE (azioni expert - azioni predette).
Una volta allenata la prima policy, le azioni intraprese sull'ambiente non saranno date solo dall'esperto ma in parte anche dalla rete:

$$
policy = a \cdot \beta + agent \cdot (1-\beta)
$$

dove "a" rappresenta le azioni dell'expert, "agent" quelle predette dalla rete e $\beta$ un coefficiente di equilibrio (inizialmente $\beta$ = 1 e quindi si hanno solo azioni dell'expert: finito il primo training $\beta$ = 0.9 e così via). 
Se si vuole eseguire un numero elevato di iterazioni, $\beta$ -> 0 e le azioni prevalenti saranno quelle predette dalla rete.

![Immagine dagger](https://github.com/MatteoMariani99/psd_DaGGER/blob/main/docs/immagini/dagger.png)

Per l'implementazione dell'algoritmo sono state prese in considerazione due tipologie di tracciato:
- **tracciato su strada**: è il tracciato del percorso di Formula1 di Barcellona-Catalogna;
  
![Immagine dagger](https://github.com/MatteoMariani99/psd_DaGGER/blob/main/docs/immagini/strada.png)
- **tracciato con coni**: è il tracciato delimitato da coni gialli e blu (si considerano una serie di circuiti della Formula Student).
  
![Immagine dagger](https://github.com/MatteoMariani99/psd_DaGGER/blob/main/docs/immagini/coni.png)

Per il **tracciato con i coni** è stato necessario utilizzare il modello [Yolov8](https://github.com/ultralytics/ultralytics?tab=readme-ov-file) per il riconoscimento e l'identificazione, tramite bounding box, all'interno dell'immagine. 

![Immagine dagger](https://github.com/MatteoMariani99/psd_DaGGER/blob/main/docs/immagini/coni_identificati.png)

## Installazione
In primis è necessario eseguire il clone della repository tramite il comando:
```bash
git clone --recurse-submodules https://github.com/MatteoMariani99/psd_DaGGER.git
```
Il tag --recurse-submodules è necessario per clonare anche il submodule relativo alla cone detection.
Il modo più semplice per poter utilizzare l'algoritmo è quello di installare [anaconda](https://www.anaconda.com/) e successivamente eseguire i seguenti comandi all'interno della repository:

```bash
conda env create -f dagger_cpu.yml
```
se si possiede una GPU Nvidia è possibile invece sfruttarne le potenzialità di calcolo creando l'ambiente:
```bash
conda env create -f dagger.yml
```
In questo modo vengono installate tutte le dipendenze necessarie per il funzionamento dell'algoritmo.
Una volta fatto ciò, l'ambiente deve essere attivato tramite il comando:
```bash
conda activate dagger_cpu
```
o
```bash
conda activate dagger
```

In [Visual Studio Code](https://code.visualstudio.com) è possibile installare l'apposita estensione **Python Environment Manager** per la gestione degli ambienti python/conda: in questo modo è possibile attivare l'ambiente che si desidera senza dover eseguire i comandi sopra riportati.


Per poter eseguire il testing basta eseguire lo script:
```bash
python3 test_agent.py
```






