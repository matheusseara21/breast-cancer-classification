import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# ==========================================
# 1. CARREGAMENTO E ANÁLISE EXPLORATÓRIA
# ==========================================
print("--- Iniciando Processamento ---")
dados = pd.read_csv('breast-cancer.csv')

# Análise exploratória inicial
print(f"Shape original: {dados.shape}")
print(f"\nPrimeiras linhas:")
print(dados.head())
print(f"\nTipos de dados:")
print(dados.dtypes)
print(f"\nValores faltantes:")
print(dados.isnull().sum())

# Tratamento de dados faltantes
dados = dados.replace('?', np.nan)
print(f"\nLinhas com valores faltantes: {dados.isnull().sum().sum()}")
dados_limpos = dados.dropna()
print(f"Linhas após limpeza: {dados_limpos.shape[0]}")

# Verificar balanceamento das classes
print(f"\nDistribuição da classe target:")
print(dados_limpos['Class'].value_counts())

# ==========================================
# 2. PRÉ-PROCESSAMENTO
# ==========================================
X = dados_limpos.drop(columns=['Class'])
y = dados_limpos['Class']

X_encoded = pd.get_dummies(X)
colunas_modelo = X_encoded.columns

print(f"\nShape após one-hot encoding: {X_encoded.shape}")
print(f"Número de features: {len(colunas_modelo)}")

# ==========================================
# 3. DIVISÃO DOS DADOS E TREINAMENTO
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, 
    test_size=0.3, 
    random_state=42,
    stratify=y
)

# Modelo com hiperparâmetros
tree_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=5
)

tree_model.fit(X_train, y_train)
print(f"\n--- Modelo Treinado ---")
print(f"Profundidade da árvore: {tree_model.get_depth()}")
print(f"Número de folhas: {tree_model.get_n_leaves()}")

# ==========================================
# 4. AVALIAÇÃO DO MODELO
# ==========================================
y_pred = tree_model.predict(X_test)

# Acurácia
acc = accuracy_score(y_test, y_pred)
print(f'\n--- Performance do Modelo ---')
print(f'Acurácia global: {acc:.2%}')

# Matriz de Confusão
cm = confusion_matrix(y_test, y_pred, labels=tree_model.classes_)
print("\nMatriz de Confusão:")
print(cm)

# Relatório de classificação
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=tree_model.classes_))

# Visualizações
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Matriz de Confusão
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=tree_model.classes_)
disp.plot(ax=ax1, cmap=plt.cm.Blues)
ax1.set_title("Matriz de Confusão")

# Feature Importance
if hasattr(tree_model, 'feature_importances_'):
    importances = tree_model.feature_importances_
    feature_names = colunas_modelo
    indices = np.argsort(importances)[::-1][:10]
    
    ax2.barh(range(len(indices)), importances[indices])
    ax2.set_yticks(range(len(indices)))
    ax2.set_yticklabels([feature_names[i] for i in indices])
    ax2.set_title("Top 10 Features Mais Importantes")
    ax2.set_xlabel("Importância")

plt.tight_layout()
plt.show()

# ==========================================
# 5. CLASSIFICAÇÃO DE NOVAS INSTÂNCIAS
# ==========================================
def classificar_nova_instancia(dados_paciente, modelo, colunas_modelo):
    """
    Classifica uma nova instância garantindo alinhamento com o modelo treinado
    """
    try:
        # Transformar em DataFrame
        nova_df = pd.DataFrame(dados_paciente)
        
        # Aplicar one-hot encoding
        nova_encoded = pd.get_dummies(nova_df)
        
        # Alinhamento de colunas
        nova_final = nova_encoded.reindex(columns=colunas_modelo, fill_value=0)
        
        # Verificar colunas faltantes
        colunas_faltantes = set(colunas_modelo) - set(nova_encoded.columns)
        if colunas_faltantes:
            print(f"Colunas adicionadas com zeros: {list(colunas_faltantes)}")
        
        # Previsão
        predicao = modelo.predict(nova_final)
        probs = modelo.predict_proba(nova_final)
        
        print("\n" + "="*50)
        print("RESULTADO DA CLASSIFICAÇÃO:")
        print("="*50)
        print(f"Classe predita: {predicao[0]}")
        print("\nProbabilidades:")
        for classe, prob in zip(modelo.classes_, probs[0]):
            print(f"  {classe}: {prob:.2%}")
        
        return predicao[0], probs[0]
    
    except Exception as e:
        print(f"Erro na classificação: {e}")
        return None, None

# Exemplo de uso
print("\n--- Classificando Nova Paciente ---")
nova_paciente = {
    'age': ['40-49'],
    'menopause': ['premeno'],
    'tumor-size': ['20-24'],
    'inv-nodes': ['0-2'],
    'node-caps': ['no'],
    'deg-malig': [2],
    'breast': ['left'],
    'breast-quad': ['left_up'],
    'irradiat': ['no']
}

classificar_nova_instancia(nova_paciente, tree_model, colunas_modelo)

# ==========================================
# 6. CLASSIFICAÇÃO DE MÚLTIPLAS INSTÂNCIAS
# ==========================================
print("\n--- Classificando Múltiplas Instâncias ---")

outros_exemplos = [
    {
        'age': ['50-59'],
        'menopause': ['ge40'],
        'tumor-size': ['30-34'],
        'inv-nodes': ['3-5'],
        'node-caps': ['yes'],
        'deg-malig': [3],
        'breast': ['right'],
        'breast-quad': ['central'],
        'irradiat': ['yes']
    }
]

for i, exemplo in enumerate(outros_exemplos, 1):
    print(f"\nExemplo {i}:")
    classificar_nova_instancia(exemplo, tree_model, colunas_modelo)