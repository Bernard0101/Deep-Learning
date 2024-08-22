    out_hidden_features = []
        
        # A matriz de pesos completa para a camada atual
        weights = self.hidden_weights[layer - 1]  # Pesos conectando a camada layer-1 à layer

        for node in range(self.hidden_nodes[layer]):
            # Pega o vetor de pesos para o nó atual na camada atual
            node_weights = weights[node]
            print(f"\nLayer:{layer}\nNode:{node}\nPredictions: {predictions}\nWeights: {node_weights}")

            # Realiza a multiplicação de predictions pela linha de pesos
            Z = np.dot(predictions, node_weights)
            out_hidden_features.append(Z)
        
        out_hidden_features = np.squeeze(out_hidden_features)
        return out_hidden_features
