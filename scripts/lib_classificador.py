import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as pl
from matplotlib.ticker import FuncFormatter
from matplotlib.cm import get_cmap
from itertools import cycle
import itertools
import urllib

from sklearn.datasets import make_classification, load_digits, load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn import manifold, datasets
from sklearn.manifold import TSNE

from deslib.util.datasets import make_P2
import xgboost as xgb

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

class UtilClassificadores:
    """
    Classe utilitária para gerenciar e avaliar classificadores de machine learning
    
    Esta classe facilita a criação, treinamento, avaliação e visualização de múltiplos
    classificadores de forma padronizada e eficiente.
    
    Atributos:
        TITLES: Lista de classificadores disponíveis
        titles: Lista de classificadores selecionados
        methods: Lista de objetos classificadores instanciados
        valid_titles: Lista de classificadores válidos após criação
        trained_models: Dicionário com modelos treinados e seus status
    """
    
    TITLES = ['DecisionTree', 'KNN', 'KNN_W', 'NaiveBayes', 'MLP', 'RF', 'XGBoost', 'ADA', 'Bag']
        
    def __init__(self, titles:list = None, arquivo:str = None):
        """
        Inicializa a classe UtilClassificadores
        
        Parâmetros:
        - titles: lista de nomes dos classificadores desejados (opcional)
        - arquivo: caminho para arquivo de classificadores salvos (opcional)
        """
        self.titles = titles if titles is not None else self.TITLES
        self.titles = [title for title in self.titles if title in self.TITLES]
        self.methods = []
        self.valid_titles = []
        self.trained_models = {}
        
        if arquivo and os.path.isfile(arquivo):
            print(f"Carregando classificadores do arquivo: {arquivo}")
            self.load_classifiers_from_file(arquivo)
        else:
            print(f"Criando classificadores: {self.titles}")
            self.methods, self.valid_titles = self.create_classifiers(self.titles)


    def load_classifiers_from_file(self, arquivo:str):
        """
        Carrega classificadores de um arquivo
        
        Parâmetros:
        - arquivo: caminho do arquivo contendo os classificadores
        """
        # TODO: Implementar carregamento de classificadores
        pass

    def save_classifiers_to_file(self, arquivo:str):
        """
        Salva classificadores em um arquivo
        
        Parâmetros:
        - arquivo: caminho do arquivo para salvar os classificadores
        """
        # TODO: Implementar salvamento de classificadores
        pass

    def get_titles_trained_models(self):
        """
        Retorna os títulos dos modelos treinados
        
        Retorna:
        - list: lista com títulos dos modelos válidos
        """
        return self.valid_titles
    
    def get_trained_models(self):
        """
        Retorna o dicionário com todos os modelos treinados
        
        Retorna:
        - dict: dicionário com modelos treinados e seus status
        """
        return self.trained_models
    
    def get_successful_models(self):
        """
        Retorna apenas os modelos treinados com sucesso
        
        Retorna:
        - dict: dicionário filtrado com modelos bem-sucedidos
        """
        return {k: v for k, v in self.trained_models.items() if v['status'] == 'success'}
    
    def __repr__(self):
        """
        Representação textual da classe
        
        Retorna:
        - str: representação formatada da classe
        """
        trained_count = len(self.trained_models)
        successful_count = len(self.get_successful_models())
        return f"UtilClassificadores(modelos={len(self.valid_titles)}, treinados={trained_count}, sucesso={successful_count})"

    def create_classifiers(self, titles=None):
        """
        Cria e configura classificadores de machine learning
        
        Parâmetros:
        - titles: lista de nomes dos classificadores desejados (opcional)
                Se None, retorna todos os classificadores disponíveis
        
        Retorna:
        - tuple (titles, methods): listas com nomes e objetos dos classificadores
        """
        ['DecisionTree', 'RF', 'KNN', 'NaiveBayes', 'XGBoost', 'ADA']
        # Definir todos os classificadores disponíveis
        available_classifiers = {
            'DecisionTree': DecisionTreeClassifier(criterion='entropy'),
            'KNN': KNeighborsClassifier(n_neighbors=3),
            'KNN_W': KNeighborsClassifier(n_neighbors=3, weights='distance'),
            'NaiveBayes': GaussianNB(var_smoothing=1e-09),
            'MLP': MLPClassifier(solver='lbfgs', early_stopping=True, hidden_layer_sizes=(32), 
                            activation='logistic', batch_size=100, max_iter=10000, 
                            learning_rate_init=0.001, momentum=0.8, random_state=46),
            'RF': RandomForestClassifier(n_estimators=100, random_state=0),
            'XGBoost': xgb.XGBClassifier(objective="binary:logistic", random_state=42),
            'ADA': AdaBoostClassifier(n_estimators=100, random_state=0),
            'Bag': BaggingClassifier(estimator=KNeighborsClassifier(n_neighbors=3), n_estimators=100, random_state=0)
        }
        
        # Configurar SVM com GridSearch
        svm_parameters = [
            {'C': [0.1, 0.5, 1, 10, 100, 500, 1000], 'kernel': ['poly']},
            {'C': [0.1, 0.5, 1, 10, 100, 500, 1000], 'gamma': [0.1, 0.001, 0.0001, 0.00001], 'kernel': ['rbf']},
        ]
        svm_base = SVC(gamma='scale', probability=True, random_state=42)
        available_classifiers['SVM'] = GridSearchCV(svm_base, svm_parameters, scoring='accuracy', cv=10)
        
        # Configurar ensemble voting
        dt_for_voting = DecisionTreeClassifier(criterion='entropy')
        nb_for_voting = GaussianNB(var_smoothing=1e-09)
        knn_for_voting = KNeighborsClassifier(n_neighbors=3)
        
        available_classifiers['DT+NB+KNN'] = VotingClassifier(
            estimators=[('DecisionTree', dt_for_voting), ('NaiveBayes', nb_for_voting), ('knn', knn_for_voting)], 
            voting='soft'
        )
        
        # Se titles não foi especificado, usar configuração padrão
        if titles is None:
            titles = ['DecisionTree', 'RF', 'KNN', 'NaiveBayes', 'XGBoost', 'ADA']
        
        # Validar se todos os classificadores solicitados existem
        methods = []
        valid_titles = []
        
        print(f"🔧 Configurando classificadores solicitados...")
        
        for title in titles:
            if title in available_classifiers:
                methods.append(available_classifiers[title])
                valid_titles.append(title)
                print(f"   ✅ {title}: Configurado")
            else:
                print(f"   ❌ {title}: Classificador não encontrado")
                print(f"      Disponíveis: {list(available_classifiers.keys())}")
        
        print(f"\n📊 Resumo:")
        print(f"   • Classificadores solicitados: {len(titles)}")
        print(f"   • Classificadores configurados: {len(valid_titles)}")
        print(f"   • Lista final: {valid_titles}")
        
        return methods, valid_titles


    def train_classifiers(self, X_train=None, y_train=None, verbose=True):
        """
        Treina classificadores de machine learning
        
        Parâmetros:
        - X_train: dados de treino (features)
        - y_train: rótulos de treino (target)
        - verbose: se True, exibe progresso detalhado
        
        Retorna:
        - dict: dicionário com modelos treinados e status
        """
        
        methods = self.methods
        titles = self.valid_titles
        
        if verbose:
            print("🔧 TREINANDO CLASSIFICADORES")
            print("=" * 50)
            print(f"📊 Configuração:")
            print(f"   • Número de classificadores: {len(methods)}")
            print(f"   • Shape treino: {X_train.shape}")
            print(f"   • Classes únicas: {len(np.unique(y_train))}")
        
        # Dicionário para armazenar os modelos treinados
        trained_models = {}

        # Treinar cada classificador
        for i, (method, name) in enumerate(zip(methods, titles)):
            print("Method:", method)
            if verbose:
                print(f"\n🔍 [{i+1}/{len(methods)}] Treinando: {name}")
            
            try:
                # Treinar o modelo
                if verbose:
                    print(f"   🎯 Treinando modelo...")
                
                method.fit(X_train, y_train)
                
                trained_models[name] = {
                    'model': method,
                    'status': 'success'
                }
                
                if verbose:
                    print(f"   ✅ {name}: Treinamento concluído com sucesso")
                    
            except Exception as e:
                if verbose:
                    print(f"   ❌ Erro ao treinar {name}: {str(e)}")
                
                trained_models[name] = {
                    'model': method,
                    'status': 'error',
                    'error': str(e)
                }
        
        if verbose:
            successful_models = {k: v for k, v in trained_models.items() if v['status'] == 'success'}
            error_models = {k: v for k, v in trained_models.items() if v['status'] == 'error'}
            
            print(f"\n📈 RESUMO DO TREINAMENTO:")
            print("=" * 40)
            print(f"✅ Modelos treinados com sucesso: {len(successful_models)}")
            if error_models:
                print(f"❌ Modelos com erro: {len(error_models)}")
                for name, data in error_models.items():
                    print(f"   • {name}: {data['error']}")
        
        self.trained_models = trained_models
        return trained_models


    def evaluate_classifiers(self, X_test_scaled, y_test, verbose=True):
        """
        Avalia classificadores treinados de machine learning
        
        Parâmetros:
        - X_test_scaled: dados de teste (features) 
        - y_test: rótulos de teste (target)
        - verbose: se True, exibe progresso detalhado
        
        Retorna:
        - tuple: (results dict, scores list) com métricas de acurácia por classificador
        """
        
        trained_models = self.trained_models
        titles = self.valid_titles
        
        if verbose:
            print("📊 AVALIANDO CLASSIFICADORES")
            print("=" * 50)
            print(f"🎯 Configuração:")
            print(f"   • Número de modelos: {len(trained_models)}")
            print(f"   • Shape teste: {X_test_scaled.shape}")
            print(f"   • Classes únicas: {len(np.unique(y_test))}")
        
        # Dicionário para armazenar os resultados
        results = {}
        scores = []
        
        # Avaliar cada classificador
        for i, name in enumerate(titles):
            if name in trained_models:
                model_info = trained_models[name]
                
                if verbose:
                    print(f"\n📊 [{i+1}/{len(titles)}] Avaliando: {name}")
                
                if model_info['status'] == 'success':
                    try:
                        # Calcular acurácia no conjunto de teste
                        model = model_info['model']
                        accuracy = model.score(X_test_scaled, y_test)
                        scores.append(accuracy)
                        results[name] = {
                            'accuracy': accuracy,
                            'model': model,
                            'status': 'success'
                        }
                        
                        if verbose:
                            print(f"   ✅ Classification accuracy {name} = {accuracy:.4f}")
                            
                    except Exception as e:
                        if verbose:
                            print(f"   ❌ Erro ao avaliar {name}: {str(e)}")
                        
                        scores.append(None)
                        results[name] = {
                            'accuracy': None,
                            'model': model_info['model'],
                            'status': 'error',
                            'error': str(e)
                        }
                else:
                    # Modelo não foi treinado com sucesso
                    if verbose:
                        print(f"   ⚠️ {name}: Modelo não foi treinado (pulando avaliação)")
                    
                    scores.append(None)
                    results[name] = {
                        'accuracy': None,
                        'model': model_info['model'],
                        'status': 'not_trained',
                        'error': model_info.get('error', 'Modelo não foi treinado com sucesso')
                    }
        
        if verbose:
            print(f"\n📈 RESUMO DOS RESULTADOS:")
            print("=" * 50)
            
            # Filtrar apenas sucessos
            successful_results = {k: v for k, v in results.items() if v['status'] == 'success'}
            
            if successful_results:
                # Ordenar por acurácia
                sorted_results = sorted(successful_results.items(), 
                                    key=lambda x: x[1]['accuracy'], 
                                    reverse=True)
                
                print(f"✅ Modelos avaliados com sucesso: {len(successful_results)}")
                print(f"📊 Ranking por acurácia:")
                
                for rank, (name, data) in enumerate(sorted_results, 1):
                    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank:2d}."
                    print(f"   {medal} {name:15s}: {data['accuracy']:7.4f}")
                
                # Estatísticas resumidas
                accuracies = [data['accuracy'] for data in successful_results.values()]
                print(f"\n📊 Estatísticas:")
                print(f"   • Melhor acurácia: {max(accuracies):.4f}")
                print(f"   • Pior acurácia: {min(accuracies):.4f}")
                print(f"   • Acurácia média: {np.mean(accuracies):.4f}")
                print(f"   • Desvio padrão: {np.std(accuracies):.4f}")
            
            # Relatório de erros
            error_results = {k: v for k, v in results.items() if v['status'] in ['error', 'not_trained']}
            if error_results:
                print(f"\n❌ Modelos com problemas: {len(error_results)}")
                for name, data in error_results.items():
                    print(f"   • {name}: {data['error']}")
        
        return results, scores

    def get_methods_trained_models(self, trained_models=None):
        """
        Extrai apenas os métodos/modelos treinados com sucesso
        
        Parâmetros:
        - trained_models: dicionário com modelos treinados (opcional, usa self.trained_models se None)
        
        Retorna:
        - list: lista de modelos treinados com sucesso
        """
        if trained_models is None:
            trained_models = self.trained_models
            
        methods = []
        valid_titles = []
        
        for title in trained_models.keys():
            if title in trained_models and trained_models[title]['status'] == 'success':
                methods.append(trained_models[title]['model'])
                valid_titles.append(title)
                
        return methods, valid_titles

    def generate_model_metrics_dataset(self, X_test_scaled, y_test, dataset_name, 
                                    save_to_csv=True, save_dir='../dataset', 
                                    display_results=True, verbose=True):
        """
        Gera dataset completo com métricas de avaliação para múltiplos modelos de ML
        
        Parâmetros:
        - X_test_scaled: dados de teste (features)
        - y_test: rótulos de teste (target)
        - dataset_name: nome do dataset para identificação
        - save_to_csv: se True, salva o dataset em arquivo CSV
        - save_dir: diretório para salvar o arquivo CSV
        - display_results: se True, exibe tabelas e estatísticas detalhadas
        - verbose: se True, exibe progresso detalhado
        
        Retorna:
        - pandas.DataFrame: DataFrame com métricas de todos os modelos
        """
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
        import pandas as pd
        import os
        from datetime import datetime
        import numpy as np
        
        methods_trained_models = self.trained_models
        titles = self.valid_titles
        
        if verbose:
            print("📊 GERANDO DATASET COM MÉTRICAS DE AVALIAÇÃO DOS MODELOS")
            print("=" * 70)
            print(f"🎯 Configuração:")
            print(f"   • Número de modelos: {len(methods_trained_models)}")
            print(f"   • Dataset: {dataset_name}")
            print(f"   • Shape dados teste: {X_test_scaled.shape}")
            print(f"   • Classes únicas: {len(np.unique(y_test))}")
            print(f"   • Salvar CSV: {'Sim' if save_to_csv else 'Não'}")
        
        # Lista para armazenar as métricas
        metrics_data = []
        
        # Filtrar apenas modelos treinados com sucesso      
        methods, valid_titles = self.get_methods_trained_models(methods_trained_models)
        print("Titles valids:", valid_titles)
        # Iterar por todos os modelos treinados
        for i, (clf, title) in enumerate(zip(methods, titles)):
            if verbose:
                print(f"\n🔍 [{i+1}/{len(methods)}] Avaliando modelo: {title}")
            
            try:
                # Fazer predições no conjunto de teste
                y_pred = clf.predict(X_test_scaled)
                
                # Calcular métricas principais
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                # Adicionar dados à lista
                metrics_data.append({
                    'modelo': title,
                    'acuracia': accuracy,
                    'precisao': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'dataset': dataset_name,
                    'data_avaliacao': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'num_amostras_teste': len(y_test),
                    'num_classes': len(np.unique(y_test)),
                    'status': 'success'
                })
                
                if verbose:
                    print(f"   ✅ {title}: Acurácia={accuracy:.4f}, Precisão={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
                    
            except Exception as e:
                if verbose:
                    print(f"   ❌ Erro ao avaliar {title}: {e}")
                
                # Adicionar dados com erro
                metrics_data.append({
                    'modelo': title,
                    'acuracia': None,
                    'precisao': None,
                    'recall': None,
                    'f1_score': None,
                    'dataset': dataset_name,
                    'data_avaliacao': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'num_amostras_teste': len(y_test),
                    'num_classes': len(np.unique(y_test)),
                    'status': 'error',
                    'erro': str(e)
                })

        # Criar DataFrame com as métricas
        metrics_df = pd.DataFrame(metrics_data)
        
        if display_results and verbose:
            # Exibir o dataset
            print(f"\n📋 DATASET DE MÉTRICAS GERADO:")
            print("=" * 80)
            display_columns = ['modelo', 'acuracia', 'precisao', 'recall', 'f1_score', 'status']
            print(metrics_df[display_columns].to_string(index=False, float_format='%.4f'))
            
            # Estatísticas resumidas
            print(f"\n📈 ESTATÍSTICAS RESUMIDAS:")
            print("=" * 50)
            
            # Filtrar apenas modelos com sucesso
            valid_metrics = metrics_df[metrics_df['acuracia'].notna()]
            if len(valid_metrics) > 0:
                print(f"✅ Modelos avaliados com sucesso: {len(valid_metrics)}")
                print(f"❌ Modelos com erro: {len(metrics_df) - len(valid_metrics)}")
                
                # Melhores resultados por métrica
                print(f"\n🏆 MELHORES RESULTADOS:")
                print(f"   • Melhor acurácia: {valid_metrics['acuracia'].max():.4f} ({valid_metrics.loc[valid_metrics['acuracia'].idxmax(), 'modelo']})")
                print(f"   • Melhor precisão: {valid_metrics['precisao'].max():.4f} ({valid_metrics.loc[valid_metrics['precisao'].idxmax(), 'modelo']})")
                print(f"   • Melhor recall: {valid_metrics['recall'].max():.4f} ({valid_metrics.loc[valid_metrics['recall'].idxmax(), 'modelo']})")
                print(f"   • Melhor F1-Score: {valid_metrics['f1_score'].max():.4f} ({valid_metrics.loc[valid_metrics['f1_score'].idxmax(), 'modelo']})")
                
                # Médias das métricas
                print(f"\n📊 MÉDIAS DAS MÉTRICAS:")
                print(f"   • Acurácia média: {valid_metrics['acuracia'].mean():.4f} (±{valid_metrics['acuracia'].std():.4f})")
                print(f"   • Precisão média: {valid_metrics['precisao'].mean():.4f} (±{valid_metrics['precisao'].std():.4f})")
                print(f"   • Recall médio: {valid_metrics['recall'].mean():.4f} (±{valid_metrics['recall'].std():.4f})")
                print(f"   • F1-Score médio: {valid_metrics['f1_score'].mean():.4f} (±{valid_metrics['f1_score'].std():.4f})")
            else:
                print("❌ Nenhum modelo foi avaliado com sucesso!")
        
        # Salvar o dataset em arquivo CSV
        if save_to_csv:
            if verbose:
                print(f"\n💾 SALVANDO DATASET:")
                print("-" * 30)
            
            try:
                # Criar diretório se não existir
                os.makedirs(save_dir, exist_ok=True)
                
                # Gerar nome do arquivo com timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                metrics_filename = f'metricas_modelos_{dataset_name}_{timestamp}.csv'
                metrics_filepath = os.path.join(save_dir, metrics_filename)
                
                # Salvar arquivo
                metrics_df.to_csv(metrics_filepath, index=False, encoding='utf-8')
                
                if verbose:
                    print(f"   ✅ Dataset salvo em: {metrics_filepath}")
                    
                    # Verificar o tamanho do arquivo
                    file_size = os.path.getsize(metrics_filepath)
                    print(f"   📁 Tamanho do arquivo: {file_size} bytes ({file_size/1024:.2f} KB)")
                    
            except Exception as e:
                if verbose:
                    print(f"   ❌ Erro ao salvar dataset: {e}")
        
        if display_results and verbose:
            # Ranking dos modelos por acurácia
            print(f"\n🏆 RANKING DOS MODELOS (por acurácia):")
            print("=" * 70)
            
            valid_models = metrics_df[metrics_df['acuracia'].notna()].copy()
            if len(valid_models) > 0:
                valid_models_sorted = valid_models.sort_values('acuracia', ascending=False).reset_index(drop=True)
                
                for i, row in valid_models_sorted.iterrows():
                    medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
                    print(f"{medal} {row['modelo']:15s}: {row['acuracia']:6.4f} " +
                        f"(Precisão: {row['precisao']:6.4f}, Recall: {row['recall']:6.4f}, F1: {row['f1_score']:6.4f})")
            
            # Listar erros se houver
            error_models = metrics_df[metrics_df['status'] == 'error']
            if len(error_models) > 0:
                print(f"\n❌ MODELOS COM ERRO:")
                print("-" * 40)
                for _, row in error_models.iterrows():
                    erro_msg = row.get('erro', 'Erro não especificado')
                    print(f"   • {row['modelo']}: {erro_msg}")
            
            print(f"\n🎉 Análise completa! Dataset gerado com {len(metrics_df)} modelos.")
            print(f"   • Sucessos: {len(valid_models)} modelos")
            print(f"   • Erros: {len(error_models)} modelos")
            print(f"💡 O DataFrame foi retornado para análises adicionais.")
        
        return metrics_df



    def generate_confusion_matrices(self, X_test_scaled, y_test, figsize_per_model=(7, 4), cols=2, save_path=None, verbose=True):
        """
        Gera matrizes de confusão para múltiplos classificadores
        
        Parâmetros:
        - X_test_scaled: dados de teste (features) já normalizados
        - y_test: rótulos de teste (target)
        - figsize_per_model: tupla com tamanho base de cada subplot (largura, altura)
        - cols: número de colunas no layout de subplots (padrão: 2)
        - save_path: caminho para salvar a figura (opcional)
        - verbose: se True, exibe progresso detalhado
        
        Retorna:
        - dict: dicionário com métricas de acurácia por classificador
        """
        
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np

        trained_models = self.trained_models
        titles = self.valid_titles
        
        # Filtrar apenas modelos treinados com sucesso
        methods, valid_titles = self.get_methods_trained_models(trained_models)
        titles = valid_titles
        
        print("=====", titles)
        if verbose:
            print("🎯 GERANDO MATRIZES DE CONFUSÃO")
            print("=" * 50)
            print(f"📊 Configuração:")
            print(f"   • Número de modelos: {len(methods)}")
            print(f"   • Layout: {cols} colunas")
            print(f"   • Shape dados teste: {X_test_scaled.shape}")
            print(f"   • Classes únicas: {len(np.unique(y_test))}")
        
        # Calcular número de linhas necessárias
        n_models = len(methods)
        n_rows = (n_models + cols - 1) // cols  # Arredonda para cima
        
        # Criar subplots
        fig_width = figsize_per_model[0] * cols
        fig_height = figsize_per_model[1] * n_rows
        
        fig, axes = plt.subplots(n_rows, cols, figsize=(fig_width, fig_height))
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        
        # Garantir que axes seja sempre um array 2D
        if n_rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Dicionário para armazenar resultados
        results = {}
        
        # Iterar pelos modelos e plotar matrizes de confusão
        for i, (clf, title) in enumerate(zip(methods, titles)):
            # Calcular posição na grade
            print(">>>>", clf, title)
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            if verbose:
                print(f"\n📊 [{i+1}/{n_models}] Processando: {title}")
            
            try:
                # Fazer predições
                y_predicted = clf.predict(X_test_scaled)
                
                # Calcular matriz de confusão
                cm = confusion_matrix(y_test, y_predicted)
                
                # Calcular acurácia
                accuracy = (cm.diagonal().sum() / cm.sum()) * 100
                
                # Criar DataFrame para melhor visualização
                unique_labels = np.unique(np.concatenate([y_test, y_predicted]))
                df_cm = pd.DataFrame(cm, 
                                index=[f'Real {label}' for label in unique_labels],
                                columns=[f'Pred {label}' for label in unique_labels])
                
                # Plotar usando seaborn heatmap
                sns.heatmap(df_cm, 
                        annot=True, 
                        fmt='d', 
                        cmap='Blues',
                        ax=ax,
                        cbar_kws={'shrink': 0.8})
                
                # Configurar subplot
                ax.set_title(f'Matriz de Confusão - {title}', 
                            fontsize=12, fontweight='bold')
                ax.set_xlabel('Classe Predita', fontsize=10)
                ax.set_ylabel('Classe Real', fontsize=10)
                
                # Adicionar texto com acurácia
                ax.text(0.02, 0.98, f'Acurácia: {accuracy:.2f}%', 
                    transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)
                
                # Armazenar resultados
                results[title] = {
                    'accuracy': accuracy / 100,  # Normalizar para [0,1]
                    'confusion_matrix': cm,
                    'predictions': y_predicted,
                    'status': 'success'
                }
                
                if verbose:
                    print(f"   ✅ {title}: Acurácia = {accuracy:.2f}%")
                    
            except Exception as e:
                if verbose:
                    print(f"   ❌ Erro ao processar {title}: {e}")
                
                # Plotar mensagem de erro
                ax.text(0.5, 0.5, f'Erro ao gerar\nmatriz para {title}', 
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=12, color='red')
                ax.set_title(f'Erro - {title}', fontsize=12, color='red')
                
                # Armazenar erro
                results[title] = {
                    'accuracy': None,
                    'confusion_matrix': None,
                    'predictions': None,
                    'status': 'error',
                    'error': str(e)
                }
        
        # Ocultar subplots extras
        for idx in range(n_models, n_rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].set_visible(False)
        
        # Configurar título geral
        plt.suptitle('Matrizes de Confusão - Comparação de Modelos', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Salvar figura se solicitado
        if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                if verbose:
                    print(f"\n💾 Figura salva em: {save_path}")
            except Exception as e:
                if verbose:
                    print(f"\n❌ Erro ao salvar figura: {e}")
        
        plt.show()
        
        if verbose:
            print(f"\n📈 RESUMO DOS RESULTADOS:")
            print("=" * 50)
            
            # Filtrar sucessos
            successful_results = {k: v for k, v in results.items() if v['status'] == 'success'}
            
            if successful_results:
                # Ranking por acurácia
                sorted_results = sorted(successful_results.items(), 
                                    key=lambda x: x[1]['accuracy'], 
                                    reverse=True)
                
                print(f"✅ Matrizes geradas com sucesso: {len(successful_results)}")
                print(f"📊 Ranking por acurácia:")
                
                for rank, (name, data) in enumerate(sorted_results, 1):
                    medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank:2d}."
                    accuracy_pct = data['accuracy'] * 100
                    print(f"   {medal} {name:15s}: {accuracy_pct:6.2f}%")
                
                # Estatísticas
                accuracies = [data['accuracy'] * 100 for data in successful_results.values()]
                print(f"\n📊 Estatísticas de Acurácia:")
                print(f"   • Melhor: {max(accuracies):.2f}%")
                print(f"   • Pior: {min(accuracies):.2f}%")
                print(f"   • Média: {np.mean(accuracies):.2f}%")
                print(f"   • Desvio: {np.std(accuracies):.2f}%")
            
            # Relatório de erros
            error_results = {k: v for k, v in results.items() if v['status'] == 'error'}
            if error_results:
                print(f"\n❌ Modelos com erro: {len(error_results)}")
                for name, data in error_results.items():
                    print(f"   • {name}: {data['error']}")
            
            print(f"\n🎉 Layout: {n_rows} linhas × {cols} colunas")
            print(f"📊 Total processado: {n_models} modelos")
        
        return results



    def generate_roc_auc_curves(self, X_test_scaled, y_test, nome_dataset='Dataset', verbose=True):
        """
        Gera curvas ROC/AUC para múltiplos classificadores treinados
        
        Parâmetros:
        - X_test_scaled: dados de teste (features) já normalizados
        - y_test: rótulos de teste (target)
        - nome_dataset: nome do dataset para título dos gráficos
        - verbose: se True, exibe progresso detalhado
        
        Retorna:
        - None
        """
        
        trained_models = self.trained_models
        titles = list(trained_models.keys())

        print("📈 Gerando Curvas ROC/AUC para todos os modelos...")

        # Verificar se é classificação binária ou multiclasse
        n_classes = len(np.unique(y_test))
    
        # Configurar cores para os gráficos
        colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

        # Lista para armazenar os valores AUC
        auc_scores = []

        # Filtrar apenas modelos treinados com sucesso
        methods, valid_titles = self.get_methods_trained_models(trained_models)

        if verbose:
            for title in titles:
                if title not in valid_titles:
                    print(f"   ⚠️ Pulando {title}: {trained_models.get(title, {}).get('error', 'Não encontrado')}")

        if n_classes == 2:
            # CLASSIFICAÇÃO BINÁRIA - ROC/AUC tradicional
            print("\n🎯 Processando classificação binária...")
            
            plt.figure(figsize=(12, 8))
            
            for clf, title, color in zip(methods, titles, colors):
                print(f"📊 Processando {title}...")
                
                try:
                    # Obter probabilidades de predição (necessário para ROC)
                    if hasattr(clf, "predict_proba"):
                        y_score = clf.predict_proba(X_test_scaled)[:, 1]
                    elif hasattr(clf, "decision_function"):
                        y_score = clf.decision_function(X_test_scaled)
                    else:
                        print(f"   ⚠️ {title}: Modelo não suporta probabilidades - usando predições binárias")
                        y_score = clf.predict(X_test_scaled)
                    
                    # Calcular curva ROC
                    fpr, tpr, _ = roc_curve(y_test, y_score)
                    roc_auc = auc(fpr, tpr)
                    auc_scores.append(roc_auc)
                    
                    # Plotar curva ROC
                    plt.plot(fpr, tpr, color=color, lw=2, 
                            label=f'{title} (AUC = {roc_auc:.3f})')
                    
                    print(f"   ✅ {title}: AUC = {roc_auc:.4f}")
                    
                except Exception as e:
                    print(f"   ❌ Erro ao processar {title}: {e}")
                    auc_scores.append(np.nan)
            
            # Adicionar linha diagonal (classificador aleatório)
            plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
                    label='Classificador Aleatório (AUC = 0.500)')
            
            # Configurar gráfico
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Taxa de Falso Positivo (1 - Especificidade)', fontsize=12)
            plt.ylabel('Taxa de Verdadeiro Positivo (Sensibilidade)', fontsize=12)
            plt.title(f'Curvas ROC - Comparação de Modelos\nDataset: {nome_dataset}', fontsize=14, fontweight='bold')
            plt.legend(loc="lower right", fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()


        # Gráfico de barras com scores AUC
        print(f"\n📊 Gerando gráfico de comparação dos scores AUC...")

        plt.figure(figsize=(12, 6))
        valid_indices = [i for i, score in enumerate(auc_scores) if not np.isnan(score)]
        valid_scores = [auc_scores[i] for i in valid_indices]
        valid_titles = [titles[i] for i in valid_indices]

        if valid_scores:
            colors_bar = plt.cm.Set3(np.linspace(0, 1, len(valid_scores)))
            bars = plt.bar(range(len(valid_scores)), valid_scores, color=colors_bar, 
                        edgecolor='black', linewidth=1)
            
            # Adicionar valores no topo das barras
            for i, (bar, score) in enumerate(zip(bars, valid_scores)):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.xlabel('Modelos', fontsize=12)
            plt.ylabel('Score AUC', fontsize=12)
            plt.title(f'Comparação de Scores AUC - Dataset: {nome_dataset}', 
                    fontsize=14, fontweight='bold')
            plt.xticks(range(len(valid_scores)), valid_titles, rotation=45, ha='right')
            plt.ylim(0, 1.1)
            plt.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.show()

        # Resumo dos resultados
        print(f"\n🏆 RESUMO DOS SCORES AUC:")
        print("=" * 60)
        auc_data = []
        for title, score in zip(titles, auc_scores):
            if not np.isnan(score):
                auc_data.append({'modelo': title, 'auc_score': score})
                print(f"✅ {title:15s}: {score:.4f}")
            else:
                print(f"❌ {title:15s}: ERROR")

        if auc_data:
            # Criar DataFrame com scores AUC
            auc_df = pd.DataFrame(auc_data)
            auc_df_sorted = auc_df.sort_values('auc_score', ascending=False).reset_index(drop=True)
            
            print(f"\n🥇 RANKING POR AUC:")
            print("-" * 40)
            for i, row in auc_df_sorted.iterrows():
                medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
                print(f"{medal} {row['modelo']:15s}: {row['auc_score']:.4f}")
            
            print(f"\n📊 ESTATÍSTICAS AUC:")
            print("-" * 30)
            print(f"Melhor AUC: {auc_df['auc_score'].max():.4f} ({auc_df.loc[auc_df['auc_score'].idxmax(), 'modelo']})")
            print(f"AUC Médio: {auc_df['auc_score'].mean():.4f}")
            print(f"Desvio Padrão: {auc_df['auc_score'].std():.4f}")
            
            # Disponibilizar DataFrame para uso posterior
            globals()['auc_df'] = auc_df
            print(f"\n💡 DataFrame 'auc_df' disponível para análises adicionais.")
