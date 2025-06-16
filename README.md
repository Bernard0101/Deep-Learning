# Documentazione del Progetto: Implementazione di Algoritmi di Rete Neurale orientati alla fisica

## Introduzione

Questo repository contiene implementazioni da zero di diversi algoritmi e architetture di rete neurale e il suo legame con legge e dataset fisici. L'obiettivo principale è comprendere i fondamenti, e il funzionamento generale di come alcuni modelli di rete neurale funzionano, dall'architetura, alla matematica alla codificazione degli algoritmi specifici ad ogni modello e all'aggiunzione della concordanza con le leggi della Fisica, rendendo i modelli piu addatti e specifici per affrontare problemi che conivolgono la fisica.

## Struttura del Progetto

### **`datasets`**

dove si incontrano tutti i datasets utilizzati per allenare i modelli, tutti di dataset sono stati creati artificialmente, e hanno soltanto un uso didattico, tutti dataset riguardano ad una legge fisica, come la legge del "Moto Retilineo Accelerato" (MRUA), o la legge di coulomb.

### **`Perceptron`**

Implementa il "Percettrone", la prima e più semplice architettura di una rete neurale che rappresenta un singolo neurone computazionale, ispirato matematicamente al neurone biologico umano. Questo modulo intende di "capire" relazioni semplici tra input e output, come la logica booleana delle porte logiche principali, e è il punto di partenza per esplorare il deep learning e le reti neurali piu complesse.

### **`Rete Neurale Multistrato`**

Fornisce un'implementazione della "Multilayer Perceptron" (MLP), che aggiunge più strati di neuroni, funzioni di attivazione e perdita e l'implementazione di algoritmi di addestramento dei pesi piu complessi, come lo "Stochastic Gradient Descent" (SGD) che utilizzano di concetti matematici come le derivate parziali per indicare i punti di miglior addestramento dei parametri del modello affinche si possa renderlo piu preciso. Questi concetti permettono al modello di "capire" relazioni piu complesse nei dati. In agginuzione a questo il modello conta anche con una parte direzionata ai concetti di (PIML) "physics informed mahcine learning" dove, per esempio, nella funzione di perdita, e possibile aggiungere una concordanza con le (ED) "Equazioni differenziabili" della fisica, con l'obbiettivo di migliorare l'accuratezza del modelo che le uttilizzera per convergere ad un risultato che sia fisicamente accettabile

### **`Tools`**

Fornisce l'implementazione di diversi algoritmi che permettono all'utente di pulire, trasformare e valutare i dati rispetto all'apprendimento dei modelli. Alcuni algoritmi come: la standardizzazione dei dati, oppure valutazione con algoritmi come la cross-validation

## Obiettivi del Progetto

1. **Comprensione Profonda:** Fornire un'esperienza pratica nello sviluppo manuale di reti neurali.
2. **Apprendimento Manuale:** Implementare funzioni di attivazione, algoritmi di ottimizzazione e tecniche di inizializzazione dei pesi senza l'uso di librerie preconfezionate.
3. **Relazione con la Fisica:** Capire qualli sono le relazione delle reti neurali con le leggi/equazioni della fisica approfondando nei concetti della
(PINN) "Physics Informed Neural Network"
4. **Dataset Sintetici:** Utilizzare dati creati manualmente per valutare le reti neurali.
5. **Modularità:** Creare una base facilmente estendibile per aggiungere nuove architetture o funzionalità.

## Stato del Progetto

| Modulo                          | Stato       |
| ------------------------------- | ----------- |
| **Tools**                       | Funzionante |
| **Perceptrone**                 | Funzionante |
| **Rete Neurale Multistrato**    | Funzionante |


## Come Utilizzare il Progetto

1. **Clona il Repository:**

   ```bash
   git clone https://github.com/Bernard0101/Deep_Learning.git
   ```

2. **Naviga nelle Cartelle:** Ogni directory contiene un modulo separato per una specifica architettura di rete neurale

3. **Esegui gli Script:**
   ```bash
   python -m modelli.
   ```

## Prossimi Passi


- **Aggiungere Esempi:** Fornire esempi pratici e visualizzazioni per dimostrare il funzionamento dei modelli.

- **Validazione:** Implementare alcuni atri tipi di validazione per i modeli di rete neurale, come: confusion matrix, ROC...

- **Aggiungere altri Ottimizzatori:** Aggiungere altri ottimizzattori oltre lo SGD, Come: Adagrad, Momentum...

- **Collegamento tra fisica e DL:** Cercare altre forme di collegare i due argomenti

---

Se trovi utile questo progetto o hai suggerimenti, sentiti libero di contribuire o lasciare un feedback nel repository GitHub!

