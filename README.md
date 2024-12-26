# Documentazione del Progetto: Implementazione di Algoritmi di Retti Neurali

## Introduzione

Questo repository contiene implementazioni manuali di diversi algoritmi e architetture di retti neurali. L'obiettivo principale è comprendere i fondamenti delle retti neurali, dall'architettura alla logica di apprendimento, utilizzando dataset sintetici (creati manualmente) invece di dati reali.

## Struttura del Progetto

Il progetto è organizzato in diverse directory, ciascuna con uno scopo specifico:

### **Funzioni/**

Contiene funzioni di base e algoritmi essenziali per il funzionamento delle retti neurali. Include:

- **Funzioni di Attivazione:** ReLU, Sigmoid, Tanh, e loro derivate.
- **Algoritmi di Ottimizzazione:** Discesa del gradiente stocastico (SGD) e altre tecniche.

### **Perceptron/**

Implementa il Perceptron, la prima e più semplice architettura di rette neurale. Questo modulo è il punto di partenza per esplorare il deep learning.

### **Rette Neurale Multistrato/**

Fornisce un'implementazione di una rette neurale multistrato (MLP), composta da più strati di neuroni. Questa architettura è progettata per modellare relazioni complesse nei dati.

### **Rette Neurale Convoluzionale/**

Dedicata alle retti neurali convoluzionali (CNN), utilizzate principalmente per l'elaborazione delle immagini. Questa parte del progetto è attualmente in fase di sviluppo.

## Obiettivi del Progetto

1. **Comprensione Profonda:** Fornire un'esperienza pratica nello sviluppo manuale di reti neurali.
2. **Apprendimento Manuale:** Implementare funzioni di attivazione, algoritmi di ottimizzazione e tecniche di inizializzazione dei pesi senza l'uso di librerie preconfezionate.
3. **Dataset Sintetici:** Utilizzare dati creati manualmente per validare le reti neurali.
4. **Modularità:** Creare una base facilmente estendibile per aggiungere nuove architetture o funzionalità.

## Stato del Progetto

| Modulo                          | Stato       |
| ------------------------------- | ----------- |
| **Funzioni**                    | Completato  |
| **Perceptron**                  | Completato  |
| **Rete Neurale Multistrato**    | Funzionante |
| **Rete Neurale Convoluzionale** | In sviluppo |

## Come Utilizzare il Progetto

1. **Clona il Repository:**

   ```bash
   git clone https://github.com/Bernard0101/Deep_Learning.git
   ```

2. **Naviga nelle Cartelle:** Ogni directory contiene un modulo separato per una specifica architettura di rette neurale.

3. **Esegui gli Script:** Gli script principali possono essere eseguiti per testare e addestrare i modelli con i dataset forniti.

## Prossimi Passi

- **Completare la CNN:** Terminare l'implementazione delle retti neurali convoluzionali.
- **Aggiungere Esempi:** Fornire esempi pratici e visualizzazioni per dimostrare il funzionamento dei modelli.
- **Documentazione Dettagliata:** Migliorare la documentazione per ogni modulo e funzione.
- **Validazione:** Implementare test e validazione con dataset sintetici più complessi.

---

Se trovi utile questo progetto o hai suggerimenti, sentiti libero di contribuire o lasciare un feedback nel repository GitHub!

