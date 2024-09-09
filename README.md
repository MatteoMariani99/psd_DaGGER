# Progetto Speciale per la Didattica


## Descrizione
L'obiettivo di questo progetto è la navigazione autonoma di una **auto da modellismo** su un tracciato di strada asfaltata e uno di coni (gialli e blu) per delimitare la corsia.
Il primo passo è stato quello di studiare lo stato dell’arte in modo da reperire quante più informazioni possibili: in particolare viene preso come riferimento il seguente articolo [Dagger article](https://arxiv.org/pdf/1011.0686) in modo da avere una linea guida per lo sviluppo e l’implementazione.


## Struttura
La repository è così strutturata:
- **Docs**: racchiude gli articoli dello stato dell'arte per lo studio del problema;
- **World**: racchiude tutti i file urdf e sdf per il caricamento dell'auto e del tracciato;
- **Scripts**: racchiude tutti i file utili al funzionamento dell'algoritmo;
- **data_test**: contiene tutte le imamgini/label ottenute (dataset aggregato);
- **dagger_models**: contiene i modelli allenati durante la procedura di training;
- **runs**: contiene i dati del training riproducibili su TensorBoard;
- **results**: contiene immagini relative al training (loss) e i video di test;

## Installazione
In primis è necessario eseguire il clone della repository tramite il comando:
```bash
git clone https://github.com/MatteoMariani99/psd_DaGGER.git
```
Il modo più semplice per poter seguire l'algoritmo è quello di installare [anaconda](https://www.anaconda.com/) e successivamente eseguire i seguenti comandi:

```bash
conda env create -f dagger_cpu.yml
```
se si possiede una GPU Nvidia è possibile invece sfruttarne le capacità creando l'ambiente:
```bash
conda env create -f dagger.yml
```
In questo modo vengono installate tutte le dipendenze necessarie per il funzionamento dell'algoritmo.
Una volta fatto ciò è possibile eseguire l'algoritmo all'interno dell'ambiente che deve essere attivato tramite il comando:
```bash
conda activate dagger_cpu
```
o
```bash
conda activate dagger
```

Per poter eseguire il testing basta eseguire lo script:
```bash
python3 test_agent.py
```



## Implementazione
L’articolo scelto descrive una policy denominata Dagger (Dataset Aggregation) un algoritmo iterativo che allena una policy deterministica basandosi sulle osservazioni ottenute dalla guida di un esperto (umano, p, mpc...).
Inizialmente si crea un dataset collezionando immagini dall'ambiente (sotto la policy del solo expert). Dopo aver collezionato N immagini, si procede con un primo training della policy per aggiornare i pesi della rete e minimizzare la loss tramite MSE (azioni expert - azioni predette).
Una volta allenata la prima policy, le azioni intraprese sull'ambiente non saranno date solo dall'esperto ma in parte anche dalla rete:

policy = a * beta + agent * (1-beta)

dove "a" rappresenta le azioni dell'expert e "agent" quelle della rete; il coefficiente beta rappresenta una sorta di peso (inizialmente beta = 1 e quindi abbiamo solo azioni dell'expert: finito il primo training beta = 0.9 e così via). 
Se si vogliono eseguire un numero elevato di iterazioni, alla fine beta sarà pari a 0 e ci saranno solo le azioni della rete a comandare l'auto.

![Immagine dagger](https://github.com/MatteoMariani99/psd_DaGGER/blob/main/docs/dagger.png)


