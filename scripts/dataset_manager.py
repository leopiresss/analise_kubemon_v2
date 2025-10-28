import pickle
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Union, Tuple


class DatasetManager:
    """
    Classe para gerenciar datasets de aprendizado de máquina.
    Contém funcionalidades para carregar, processar, escalar e salvar datasets
    com as variáveis necessárias para treinamento de modelos ML.
    """
    
    def __init__(self, nome_dataset: str = 'default'):
        """
        Inicializa o gerenciador de dataset.
        
        Args:
            nome_dataset (str): Nome identificador do dataset
        """
        self.nome_dataset = nome_dataset
        self.dataset = {}
        self._reset_dataset()
    
    def _reset_dataset(self):
        """Reinicializa o dataset com estrutura padrão."""
        self.dataset = {
            'nome_dataset': self.nome_dataset,
            'X_train': None,
            'X_val': None,
            'X_test': None,
            'y_train': None,
            'y_val': None,
            'y_test': None,
            'X_train_scaled': None,
            'X_val_scaled': None,
            'X_test_scaled': None,
            'features': None,
            'scaler': None,
            'label_encoder': None,
            'classes_mapping': None
        }
    
    def carregar_dados_csv(self, 
                          arquivo_csv: str, 
                          target_column: str,
                          colunas_excluir: Optional[List[str]] = None,
                          test_size: float = 0.6,
                          val_size: float = 0.5,
                          random_state: int = 42) -> Dict:
        """
        Carrega dados de um arquivo CSV e prepara para ML.
        
        Args:
            arquivo_csv (str): Caminho para o arquivo CSV
            target_column (str): Nome da coluna target
            colunas_excluir (List[str], optional): Colunas a excluir do dataset
            test_size (float): Proporção do conjunto de teste (padrão: 0.3)
            val_size (float): Proporção da validação dentro do teste (padrão: 0.5)
            random_state (int): Seed para reprodutibilidade
            
        Returns:
            Dict: Dataset estruturado com todas as variáveis
        """
        try:
            print(f"🔧 Carregando dados do arquivo: {arquivo_csv}")
            df = pd.read_csv(arquivo_csv)
            
            # Configurar colunas a excluir
            if colunas_excluir is None:
                colunas_excluir = ['os_timestamp', 'node_name', 'iteration']
            
            # Adicionar target à lista de exclusão para features
            colunas_excluir_features = colunas_excluir + [target_column]
            colunas_excluir_features = [col for col in colunas_excluir_features if col in df.columns]
            
            # Selecionar apenas features numéricas
            features_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
            features_para_modelo = [col for col in features_numericas if col not in colunas_excluir_features]
            
            print(f"📊 Informações do dataset:")
            print(f"   • Total de registros: {len(df):,}")
            print(f"   • Total de features: {len(features_para_modelo)}")
            print(f"   • Colunas excluídas: {colunas_excluir_features}")
            
            # Preparar X e y
            X = df[features_para_modelo].copy()
            y = df[target_column].copy()
            
            # Tratar valores ausentes
            valores_ausentes = X.isnull().sum().sum()
            if valores_ausentes > 0:
                print(f"   • Preenchendo {valores_ausentes:,} valores ausentes com a mediana...")
                X = X.fillna(X.median())
            
            # Codificar target se necessário
            le = LabelEncoder()
            if y.dtype == 'object':
                y_encoded = le.fit_transform(y)
                classes_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                print(f"   • Target codificado: {classes_mapping}")
                self.dataset['label_encoder'] = le
                self.dataset['classes_mapping'] = classes_mapping
            else:
                y_encoded = y.values
                self.dataset['classes_mapping'] = None
            
            # Dividir os dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded,
                test_size=test_size,
                random_state=random_state,
                stratify=y_encoded
            )
            
            # Dividir teste e validação
            X_test, X_val, y_test, y_val = train_test_split(
                X_test, y_test,
                test_size=val_size,
                random_state=random_state,
                stratify=y_test
            )
            
            # Atualizar dataset
            self.dataset.update({
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test,
                'features': features_para_modelo
            })
            
            print(f"\n✅ Dados preparados:")
            print(f"   • X_train shape: {X_train.shape}")
            print(f"   • X_val shape: {X_val.shape}")
            print(f"   • X_test shape: {X_test.shape}")
            print(f"   • Classes únicas: {np.unique(y_encoded)}")
            
            return self.dataset
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            raise
    
    def aplicar_escalonamento(self, 
                            metodo: str = 'standard',
                            salvar_automaticamente: bool = True) -> Dict:
        """
        Aplica escalonamento aos dados de treino, validação e teste.
        
        Args:
            metodo (str): Método de escalonamento ('standard', 'yeo-johnson')
            salvar_automaticamente (bool): Se deve salvar automaticamente após escalar
            
        Returns:
            Dict: Dataset com dados escalados
        """
        if self.dataset['X_train'] is None:
            raise ValueError("Dataset não carregado. Execute carregar_dados_csv() primeiro.")
        
        try:
            print(f"⚖️ Aplicando escalonamento: {metodo}")
            
            if metodo == 'standard':
                scaler = StandardScaler()
            elif metodo == 'yeo-johnson':
                scaler = PowerTransformer(method='yeo-johnson', standardize=True)
            else:
                raise ValueError(f"Método de escalonamento '{metodo}' não suportado. Use 'standard' ou 'yeo-johnson'.")
            
            # Fit apenas no conjunto de treino para evitar data leakage
            X_train_scaled = scaler.fit_transform(self.dataset['X_train'])
            X_val_scaled = scaler.transform(self.dataset['X_val'])
            X_test_scaled = scaler.transform(self.dataset['X_test'])
            
            # Atualizar dataset
            self.dataset.update({
                'X_train_scaled': X_train_scaled,
                'X_val_scaled': X_val_scaled,
                'X_test_scaled': X_test_scaled,
                'scaler': scaler
            })
            
            # Estatísticas do escalonamento
            print(f"   ✅ Escalonamento '{metodo}' aplicado com sucesso")
            print(f"   • Treino - Média antes: {self.dataset['X_train'].mean().mean():.3f} | depois: {X_train_scaled.mean():.3f}")
            print(f"   • Treino - Std antes: {self.dataset['X_train'].std().mean():.3f} | depois: {X_train_scaled.std():.3f}")
            print(f"   • Validação - Média: {X_val_scaled.mean():.3f} | Std: {X_val_scaled.std():.3f}")
            print(f"   • Teste - Média: {X_test_scaled.mean():.3f} | Std: {X_test_scaled.std():.3f}")
            
            if salvar_automaticamente:
                self.salvar_dataset()
            
            return self.dataset
            
        except Exception as e:
            print(f"❌ Erro ao aplicar escalonamento: {e}")
            raise
    
    def salvar_dataset(self, arquivo_pkl: Optional[str] = None) -> str:
        """
        Salva o dataset em arquivo pickle.
        
        Args:
            arquivo_pkl (str, optional): Caminho do arquivo. Se None, usa nome padrão.
            
        Returns:
            str: Caminho do arquivo salvo
        """
        try:
            if arquivo_pkl is None:
                arquivo_pkl = f"../datasets/{self.nome_dataset}.pkl"
            
            # Criar diretório se não existir
            os.makedirs(os.path.dirname(arquivo_pkl), exist_ok=True)
            
            with open(arquivo_pkl, 'wb') as f:
                pickle.dump(self.dataset, f)
            
            print(f"✅ Dataset salvo com sucesso em {arquivo_pkl}")
            return arquivo_pkl
            
        except Exception as e:
            print(f"❌ Erro ao salvar dataset: {e}")
            raise
    
    def carregar_dataset(self, arquivo_pkl: str) -> Dict:
        """
        Carrega dataset de arquivo pickle.
        
        Args:
            arquivo_pkl (str): Caminho do arquivo pickle
            
        Returns:
            Dict: Dataset carregado
        """
        try:
            with open(arquivo_pkl, 'rb') as f:
                self.dataset = pickle.load(f)
            
            print(f"✅ Dataset carregado de {arquivo_pkl}")
            self.nome_dataset = self.dataset.get('nome_dataset', 'loaded')
            return self.dataset
            
        except FileNotFoundError:
            print(f"❌ Arquivo não encontrado: {arquivo_pkl}")
            raise
        except Exception as e:
            print(f"❌ Erro ao carregar dataset: {e}")
            raise
    
    def aplicar_selecao_features(self, 
                               features_selecionadas: List[str],
                               salvar_automaticamente: bool = True) -> Dict:
        """
        Aplica seleção de features ao dataset.
        
        Args:
            features_selecionadas (List[str]): Lista de features selecionadas
            salvar_automaticamente (bool): Se deve salvar automaticamente
            
        Returns:
            Dict: Dataset com features selecionadas
        """
        if self.dataset['X_train'] is None:
            raise ValueError("Dataset não carregado.")
        
        try:
            print(f"🎯 Aplicando seleção de features: {len(features_selecionadas)} features")
            
            # Verificar se as features existem
            features_disponiveis = list(self.dataset['X_train'].columns)
            features_invalidas = [f for f in features_selecionadas if f not in features_disponiveis]
            
            if features_invalidas:
                raise ValueError(f"Features inválidas: {features_invalidas}")
            
            # Aplicar seleção
            self.dataset['X_train'] = self.dataset['X_train'][features_selecionadas]
            self.dataset['X_val'] = self.dataset['X_val'][features_selecionadas]
            self.dataset['X_test'] = self.dataset['X_test'][features_selecionadas]
            self.dataset['features'] = features_selecionadas
            
            # Aplicar aos dados escalados se existirem
            if self.dataset['X_train_scaled'] is not None:
                # Para dados escalados (arrays numpy), usar indexação por posição
                indices = [features_disponiveis.index(f) for f in features_selecionadas]
                self.dataset['X_train_scaled'] = self.dataset['X_train_scaled'][:, indices]
                self.dataset['X_val_scaled'] = self.dataset['X_val_scaled'][:, indices]
                self.dataset['X_test_scaled'] = self.dataset['X_test_scaled'][:, indices]
            
            print(f"   ✅ Seleção aplicada:")
            print(f"   • Novo shape X_train: {self.dataset['X_train'].shape}")
            print(f"   • Features selecionadas: {len(features_selecionadas)}")
            
            if salvar_automaticamente:
                self.salvar_dataset()
            
            return self.dataset
            
        except Exception as e:
            print(f"❌ Erro ao aplicar seleção de features: {e}")
            raise
    
    def imprimir_informacoes(self):
        """Imprime informações detalhadas do dataset."""
        print(f"\n📊 Informações do Dataset: {self.nome_dataset}")
        print("=" * 50)
        
        for key, value in self.dataset.items():
            if key == 'nome_dataset':
                print(f"Nome do dataset: {value}")
            elif key in ['X_train', 'X_val', 'X_test']:
                shape = value.shape if value is not None else None
                print(f"{key}.shape: {shape}")
            elif key in ['X_train_scaled', 'X_val_scaled', 'X_test_scaled']:
                shape = value.shape if value is not None else None
                print(f"{key}.shape: {shape}")
            elif key in ['y_train', 'y_val', 'y_test']:
                shape = len(value) if value is not None else None
                print(f"{key} length: {shape}")
            elif key == 'features':
                qtd = len(value) if value is not None else None
                print(f"Features count: {qtd}")
            elif key == 'scaler':
                tipo = type(value).__name__ if value is not None else None
                print(f"Scaler type: {tipo}")
            elif key == 'classes_mapping':
                print(f"Classes mapping: {value}")
        
        print("=" * 50)
    
    def get_data_for_training(self, usar_dados_escalados: bool = True) -> Tuple:
        """
        Retorna dados formatados para treinamento.
        
        Args:
            usar_dados_escalados (bool): Se deve usar dados escalados
            
        Returns:
            Tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if usar_dados_escalados and self.dataset['X_train_scaled'] is not None:
            return (
                self.dataset['X_train_scaled'],
                self.dataset['X_val_scaled'],
                self.dataset['X_test_scaled'],
                self.dataset['y_train'],
                self.dataset['y_val'],
                self.dataset['y_test']
            )
        else:
            return (
                self.dataset['X_train'].values,
                self.dataset['X_val'].values,
                self.dataset['X_test'].values,
                self.dataset['y_train'],
                self.dataset['y_val'],
                self.dataset['y_test']
            )
    
    def get_dataset(self) -> Dict:
        """
        Retorna o dataset completo.
        
        Returns:
            Dict: Dataset completo
        """
        return self.dataset.copy()
    
    def analisar_features_mutual_information(self, 
                                           threshold_mutual_info: float = 0.01,
                                           threshold_percentile: float = 50.0,
                                           aplicar_selecao: bool = True,
                                           metodo_selecao: str = 'combined') -> Dict:
        """
        Realiza análise de features usando Mutual Information.
        
        Args:
            threshold_mutual_info (float): Threshold mínimo para mutual information
            threshold_percentile (float): Percentil para seleção automática (0-100)
            aplicar_selecao (bool): Se deve aplicar automaticamente a seleção de features
            metodo_selecao (str): 'absolute', 'percentile' ou 'combined'
            
        Returns:
            Dict: Resultados da análise de mutual information
        """
        if self.dataset['X_train'] is None or self.dataset['y_train'] is None:
            raise ValueError("Dataset não carregado. Execute carregar_dados_csv() primeiro.")
        
        try:
            # Importar FeatureAnalyzer
            from feature_analyzer import FeatureAnalyzer
            
            print(f"🧠 Iniciando análise de features com Mutual Information...")
            
            # Criar analisador
            analyzer = FeatureAnalyzer(
                threshold_mutual_info=threshold_mutual_info,
                threshold_percentile=threshold_percentile,
                random_state=42
            )
            
            # Executar análise no conjunto de treino
            resultados = analyzer.calcular_mutual_information(
                self.dataset['X_train'], 
                self.dataset['y_train']
            )
            
            # Armazenar o analisador no dataset
            self.dataset['feature_analyzer'] = analyzer
            self.dataset['mutual_info_results'] = resultados
            
            # Aplicar seleção de features se solicitado
            if aplicar_selecao:
                features_selecionadas = analyzer.selecionar_features_por_threshold(metodo=metodo_selecao)
                
                if len(features_selecionadas) > 0:
                    print(f"🎯 Aplicando seleção automática de features...")
                    self.aplicar_selecao_features(features_selecionadas, salvar_automaticamente=False)
                    print(f"   ✅ {len(features_selecionadas)} features selecionadas aplicadas ao dataset")
                else:
                    print(f"⚠️ Nenhuma feature atendeu aos critérios de seleção")
            
            return resultados
            
        except ImportError:
            print(f"❌ Erro: Classe FeatureAnalyzer não encontrada. Verifique se o arquivo feature_analyzer.py está no mesmo diretório.")
            raise
        except Exception as e:
            print(f"❌ Erro na análise de features: {e}")
            raise
    
    def plotar_analise_features(self, top_n: int = 20):
        """
        Plota gráficos da análise de mutual information.
        
        Args:
            top_n (int): Número de top features a mostrar
        """
        if 'feature_analyzer' not in self.dataset or self.dataset['feature_analyzer'] is None:
            raise ValueError("Execute analisar_features_mutual_information() primeiro.")
        
        analyzer = self.dataset['feature_analyzer']
        
        # Plotar mutual information
        analyzer.plotar_mutual_information(top_n=top_n)
        
        # Plotar distribuição
        analyzer.plotar_distribuicao_scores()
    
    def imprimir_relatorio_features(self):
        """Imprime relatório detalhado da análise de features."""
        if 'feature_analyzer' not in self.dataset or self.dataset['feature_analyzer'] is None:
            raise ValueError("Execute analisar_features_mutual_information() primeiro.")
        
        analyzer = self.dataset['feature_analyzer']
        analyzer.imprimir_relatorio()
    
    def get_feature_ranking(self) -> pd.DataFrame:
        """
        Retorna ranking das features por mutual information.
        
        Returns:
            DataFrame: Ranking das features
        """
        if 'feature_analyzer' not in self.dataset or self.dataset['feature_analyzer'] is None:
            raise ValueError("Execute analisar_features_mutual_information() primeiro.")
        
        analyzer = self.dataset['feature_analyzer']
        return analyzer.get_feature_ranking()
    
    def analisar_features_por_ranges(self):
        """Executa análise de features por ranges de importância."""
        if 'feature_analyzer' not in self.dataset or self.dataset['feature_analyzer'] is None:
            raise ValueError("Execute analisar_features_mutual_information() primeiro.")
        
        analyzer = self.dataset['feature_analyzer']
        return analyzer.analisar_features_por_ranges()
    
    def imprimir_analise_ranges(self):
        """Imprime relatório da análise por ranges de importância."""
        if 'feature_analyzer' not in self.dataset or self.dataset['feature_analyzer'] is None:
            raise ValueError("Execute analisar_features_mutual_information() primeiro.")
        
        analyzer = self.dataset['feature_analyzer']
        analyzer.imprimir_analise_ranges()
    
    def plotar_analise_ranges(self, figsize: Tuple[int, int] = (14, 10)):
        """Plota visualizações da análise por ranges de importância."""
        if 'feature_analyzer' not in self.dataset or self.dataset['feature_analyzer'] is None:
            raise ValueError("Execute analisar_features_mutual_information() primeiro.")
        
        analyzer = self.dataset['feature_analyzer']
        analyzer.plotar_analise_ranges(figsize=figsize)
    
    def selecionar_features_por_categoria(self, categorias: List[str]) -> List[str]:
        """
        Seleciona features baseado nas categorias de importância.
        
        Args:
            categorias (List[str]): Categorias desejadas ('Muito Baixa', 'Baixa', 'Moderada', 'Alta', 'Muito Alta')
            
        Returns:
            List[str]: Features selecionadas das categorias especificadas
        """
        if 'feature_analyzer' not in self.dataset or self.dataset['feature_analyzer'] is None:
            raise ValueError("Execute analisar_features_mutual_information() primeiro.")
        
        analyzer = self.dataset['feature_analyzer']
        return analyzer.selecionar_features_por_categoria(categorias)


def main():
    """
    Função de exemplo demonstrando o uso da classe DatasetManager.
    """
    print("🚀 Demonstração da classe DatasetManager")
    print("=" * 50)
    
    # Exemplo de uso
    try:
        # Criar instância do gerenciador
        dm = DatasetManager('exemplo_svm')
        
        # Carregar dados (substitua pelo seu arquivo CSV)
        # dm.carregar_dados_csv('../datasets/svm.csv', 'target')
        
        # Aplicar escalonamento
        # dm.aplicar_escalonamento('yeo-johnson')
        
        # Exemplo de seleção de features
        # features_selecionadas = dm.dataset['features'][:10]  # Primeiras 10 features
        # dm.aplicar_selecao_features(features_selecionadas)
        
        # Imprimir informações
        # dm.imprimir_informacoes()
        
        # Obter dados para treinamento
        # X_train, X_val, X_test, y_train, y_val, y_test = dm.get_data_for_training()
        
        print("✅ Classe DatasetManager pronta para uso!")
        print("\n📝 Exemplo de uso:")
        print("dm = DatasetManager('meu_dataset')")
        print("dm.carregar_dados_csv('arquivo.csv', 'target')")
        print("dm.aplicar_escalonamento('yeo-johnson')")
        print("dm.imprimir_informacoes()")
        
    except Exception as e:
        print(f"⚠️ Exemplo não executado completamente: {e}")
        print("   Forneça um arquivo CSV válido para testar completamente.")


if __name__ == "__main__":
    main()