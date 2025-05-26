# Documentazione del Progetto: Implementazione di Algoritmi di Rete Neurale

## Introduzione

Questo repository contiene implementazioni da zero di diversi algoritmi e architetture di rete neurale. L'obiettivo principale è comprendere i fondamenti, e il funzionamento generale di come alcuni modelli di rete neurale funzionano, dall'architetura, alla matematica alla codificazione degli algoritmi specifici ad ogni modello.

## Struttura del Progetto

### **`datasets`**

dove si incontrano tutti i datasets utilizzati per allenare i modelli sia i dataset non puliti (raw) che quelli puliti

### **`Perceptron`**

Implementa il "Perceptrone", la prima e più semplice architettura di una rete neurale che rappresenta un singolo neurone computazionale, ispirato matematicamente al neurone biologico umano. Questo modulo intende di "capire" relazioni semplici tra input e output come la logica booleana, e è il punto di partenza per esplorare il deep learning e le reti neurali piu complesse.

### **`Rete Neurale Multistrato`**

Fornisce un'implementazione della "Multilayer Perceptron" (MLP), che aggiunge più strati di neuroni, funzioni di attivazione e perdita e l'implementazione di algoritmi di addestramento dei pesi piu complessi, come lo "Stochastic Gradient Descent" (SGD) che utilizzano di concetti matematici come le derivate, per indicare i punti di miglior aggiornamento dei parametri del modello per renderlo piu preciso. Questi concetti permettono al modello di "capire" relazioni piu complesse nei dati. 

### **`Tools`**

Fornisce l'implementazione di diversi algoritmi che permettono all'utente di pulire i dati e anche valutare l'apprendimento dei modelli, con algoritmi come la standardizzazione dei dati, oppure algoritmi come la cross-validation

## Obiettivi del Progetto

1. **Comprensione Profonda:** Fornire un'esperienza pratica nello sviluppo manuale di reti neurali.
2. **Apprendimento Manuale:** Implementare funzioni di attivazione, algoritmi di ottimizzazione e tecniche di inizializzazione dei pesi senza l'uso di librerie preconfezionate.
3. **Dataset Sintetici:** Utilizzare dati creati manualmente per validare le reti neurali.
4. **Modularità:** Creare una base facilmente estendibile per aggiungere nuove architetture o funzionalità.

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
   python -m src/...
   ```

## Prossimi Passi


- **Aggiungere Esempi:** Fornire esempi pratici e visualizzazioni per dimostrare il funzionamento dei modelli.

- **Validazione:** Implementare alcuni atri tipi di validazione per i modeli di rete neurale, come: confusion matrix, ROC...

- **Aggiungere altri Ottimizzatori:** aggiungere altri ottimizzattori oltre lo SGD, Come: Adagrad, Momentum...

---

Se trovi utile questo progetto o hai suggerimenti, sentiti libero di contribuire o lasciare un feedback nel repository GitHub!

