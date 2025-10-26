import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Optional, Union, Tuple
import warnings
warnings.filterwarnings('ignore')


class FeatureAnalyzer:
    """
    Classe para análise de features usando Mutual Information.
    Permite configurar thresholds e realizar seleção de features
    baseada na relevância para o target.
    """
    
    def __init__(self, threshold_mutual_info: float = 0.01, 
                 threshold_percentile: float = 50.0,
                 random_state: int = 42):
        """
        Inicializa o analisador de features.
        
        Args:
            threshold_mutual_info (float): Threshold mínimo para mutual information
            threshold_percentile (float): Percentil para seleção automática (0-100)
            random_state (int): Seed para reprodutibilidade
        """
        self.threshold_mutual_info = threshold_mutual_info
        self.threshold_percentile = threshold_percentile
        self.random_state = random_state
        
        # Resultados da análise
        self.resultados = {}
        self.features_selecionadas = None
        self.scores_mutual_info = None
        self.is_classification = None
        
    def calcular_mutual_information(self, 
                                  X: Union[pd.DataFrame, np.ndarray], 
                                  y: Union[pd.Series, np.ndarray],
                                  tipo_problema: str = 'auto') -> Dict:
        """
        Calcula mutual information entre features e target.
        
        Args:
            X (DataFrame/ndarray): Features
            y (Series/ndarray): Target
            tipo_problema (str): 'classification', 'regression' ou 'auto'
            
        Returns:
            Dict: Resultados da análise com scores e rankings
        """
        try:
            print(f"🧠 Calculando Mutual Information...")
            
            # Converter para DataFrame se necessário
            if isinstance(X, np.ndarray):
                X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
            
            # Determinar tipo do problema
            if tipo_problema == 'auto':
                unique_values = len(np.unique(y))
                self.is_classification = unique_values < 20  # Heurística simples
                tipo_problema = 'classification' if self.is_classification else 'regression'
            else:
                self.is_classification = (tipo_problema == 'classification')
            
            print(f"   • Tipo de problema detectado: {tipo_problema}")
            print(f"   • Número de features: {X.shape[1]}")
            print(f"   • Número de amostras: {X.shape[0]}")
            
            # Calcular mutual information
            if self.is_classification:
                mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
            else:
                mi_scores = mutual_info_regression(X, y, random_state=self.random_state)
            
            # Criar DataFrame com resultados
            feature_names = X.columns.tolist() if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
            
            df_results = pd.DataFrame({
                'feature': feature_names,
                'mutual_info_score': mi_scores,
                'rank': range(1, len(mi_scores) + 1)
            }).sort_values('mutual_info_score', ascending=False).reset_index(drop=True)
            
            # Atualizar rankings
            df_results['rank'] = range(1, len(df_results) + 1)
            
            # Aplicar thresholds
            threshold_value = np.percentile(mi_scores, self.threshold_percentile)
            df_results['selected_by_percentile'] = df_results['mutual_info_score'] >= threshold_value
            df_results['selected_by_threshold'] = df_results['mutual_info_score'] >= self.threshold_mutual_info
            df_results['selected_combined'] = df_results['selected_by_percentile'] & df_results['selected_by_threshold']
            
            # Armazenar resultados
            self.scores_mutual_info = mi_scores
            self.resultados = {
                'df_analysis': df_results,
                'statistics': {
                    'mean_mi_score': np.mean(mi_scores),
                    'std_mi_score': np.std(mi_scores),
                    'max_mi_score': np.max(mi_scores),
                    'min_mi_score': np.min(mi_scores),
                    'threshold_percentile_value': threshold_value,
                    'threshold_mutual_info_value': self.threshold_mutual_info,
                    'total_features': len(feature_names),
                    'selected_by_percentile': df_results['selected_by_percentile'].sum(),
                    'selected_by_threshold': df_results['selected_by_threshold'].sum(),
                    'selected_combined': df_results['selected_combined'].sum()
                },
                'tipo_problema': tipo_problema
            }
            
            # Features selecionadas (usando critério combinado por padrão)
            self.features_selecionadas = df_results[df_results['selected_combined']]['feature'].tolist()
            
            print(f"   ✅ Análise concluída:")
            print(f"   • Média MI: {self.resultados['statistics']['mean_mi_score']:.4f}")
            print(f"   • Threshold percentil {self.threshold_percentile}%: {threshold_value:.4f}")
            print(f"   • Features selecionadas (combinado): {len(self.features_selecionadas)}")
            
            return self.resultados
            
        except Exception as e:
            print(f"❌ Erro ao calcular mutual information: {e}")
            raise
    
    def selecionar_features_por_threshold(self, 
                                        threshold: float = None,
                                        metodo: str = 'absolute') -> List[str]:
        """
        Seleciona features baseado em threshold específico.
        
        Args:
            threshold (float): Valor do threshold (usa o padrão da classe se None)
            metodo (str): 'absolute', 'percentile' ou 'combined'
            
        Returns:
            List[str]: Lista de features selecionadas
        """
        if self.resultados is None:
            raise ValueError("Execute calcular_mutual_information() primeiro.")
        
        df_analysis = self.resultados['df_analysis']
        
        if threshold is not None:
            if metodo == 'absolute':
                features_selecionadas = df_analysis[df_analysis['mutual_info_score'] >= threshold]['feature'].tolist()
            elif metodo == 'percentile':
                threshold_value = np.percentile(df_analysis['mutual_info_score'], threshold)
                features_selecionadas = df_analysis[df_analysis['mutual_info_score'] >= threshold_value]['feature'].tolist()
            else:
                raise ValueError("Método deve ser 'absolute' ou 'percentile'")
        else:
            # Usar configuração da classe
            if metodo == 'absolute':
                features_selecionadas = df_analysis[df_analysis['selected_by_threshold']]['feature'].tolist()
            elif metodo == 'percentile':
                features_selecionadas = df_analysis[df_analysis['selected_by_percentile']]['feature'].tolist()
            elif metodo == 'combined':
                features_selecionadas = df_analysis[df_analysis['selected_combined']]['feature'].tolist()
            else:
                raise ValueError("Método deve ser 'absolute', 'percentile' ou 'combined'")
        
        print(f"🎯 Features selecionadas por {metodo}: {len(features_selecionadas)}")
        return features_selecionadas
    
    def selecionar_top_features(self, n_features: int) -> List[str]:
        """
        Seleciona as top N features por mutual information.
        
        Args:
            n_features (int): Número de features a selecionar
            
        Returns:
            List[str]: Lista das top N features
        """
        if self.resultados is None:
            raise ValueError("Execute calcular_mutual_information() primeiro.")
        
        df_analysis = self.resultados['df_analysis']
        top_features = df_analysis.head(n_features)['feature'].tolist()
        
        print(f"🏆 Top {n_features} features selecionadas:")
        for i, feature in enumerate(top_features, 1):
            score = df_analysis[df_analysis['feature'] == feature]['mutual_info_score'].iloc[0]
            print(f"   {i:2d}. {feature}: {score:.4f}")
        
        return top_features
    
    def plotar_mutual_information(self, 
                                top_n: int = 20,
                                figsize: Tuple[int, int] = (12, 8),
                                salvar_grafico: str = None):
        """
        Plota gráfico de barras com mutual information scores.
        
        Args:
            top_n (int): Número de top features a mostrar
            figsize (tuple): Tamanho da figura
            salvar_grafico (str): Caminho para salvar o gráfico
        """
        if self.resultados is None:
            raise ValueError("Execute calcular_mutual_information() primeiro.")
        
        df_analysis = self.resultados['df_analysis']
        df_plot = df_analysis.head(top_n)
        
        plt.figure(figsize=figsize)
        
        # Cores baseadas na seleção
        cores = ['green' if selected else 'lightgray' 
                for selected in df_plot['selected_combined']]
        
        bars = plt.barh(range(len(df_plot)), df_plot['mutual_info_score'], color=cores)
        
        plt.yticks(range(len(df_plot)), df_plot['feature'])
        plt.xlabel('Mutual Information Score')
        plt.title(f'Top {top_n} Features - Mutual Information Analysis\n'
                 f'(Problema: {self.resultados["tipo_problema"]})')
        
        # Adicionar linhas de threshold
        plt.axvline(x=self.threshold_mutual_info, color='red', linestyle='--', 
                   label=f'Threshold MI: {self.threshold_mutual_info:.3f}')
        plt.axvline(x=self.resultados['statistics']['threshold_percentile_value'], 
                   color='orange', linestyle='--', 
                   label=f'Percentil {self.threshold_percentile}%: {self.resultados["statistics"]["threshold_percentile_value"]:.3f}')
        
        plt.legend()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Adicionar anotações nas barras
        for i, (bar, score) in enumerate(zip(bars, df_plot['mutual_info_score'])):
            plt.text(score + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', va='center', fontsize=8)
        
        if salvar_grafico:
            plt.savefig(salvar_grafico, dpi=300, bbox_inches='tight')
            print(f"📊 Gráfico salvo em: {salvar_grafico}")
        
        plt.show()
    
    def plotar_distribuicao_scores(self, figsize: Tuple[int, int] = (10, 6)):
        """
        Plota distribuição dos mutual information scores.
        
        Args:
            figsize (tuple): Tamanho da figura
        """
        if self.resultados is None:
            raise ValueError("Execute calcular_mutual_information() primeiro.")
        
        scores = self.resultados['df_analysis']['mutual_info_score']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Histograma
        ax1.hist(scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=self.threshold_mutual_info, color='red', linestyle='--', 
                   label=f'Threshold MI: {self.threshold_mutual_info:.3f}')
        ax1.axvline(x=self.resultados['statistics']['threshold_percentile_value'], 
                   color='orange', linestyle='--', 
                   label=f'Percentil {self.threshold_percentile}%')
        ax1.set_xlabel('Mutual Information Score')
        ax1.set_ylabel('Frequência')
        ax1.set_title('Distribuição dos MI Scores')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Boxplot
        ax2.boxplot(scores, vert=True)
        ax2.axhline(y=self.threshold_mutual_info, color='red', linestyle='--', 
                   label=f'Threshold MI: {self.threshold_mutual_info:.3f}')
        ax2.axhline(y=self.resultados['statistics']['threshold_percentile_value'], 
                   color='orange', linestyle='--', 
                   label=f'Percentil {self.threshold_percentile}%')
        ax2.set_ylabel('Mutual Information Score')
        ax2.set_title('Boxplot dos MI Scores')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def imprimir_relatorio(self):
        """Imprime relatório detalhado da análise."""
        if self.resultados is None:
            raise ValueError("Execute calcular_mutual_information() primeiro.")
        
        stats = self.resultados['statistics']
        
        print(f"\n📊 RELATÓRIO DE ANÁLISE DE FEATURES - MUTUAL INFORMATION")
        print("=" * 70)
        print(f"🎯 Tipo de Problema: {self.resultados['tipo_problema'].upper()}")
        print(f"📈 Total de Features: {stats['total_features']}")
        print(f"\n📊 Estatísticas dos MI Scores:")
        print(f"   • Média: {stats['mean_mi_score']:.6f}")
        print(f"   • Desvio Padrão: {stats['std_mi_score']:.6f}")
        print(f"   • Máximo: {stats['max_mi_score']:.6f}")
        print(f"   • Mínimo: {stats['min_mi_score']:.6f}")
        
        print(f"\n🎯 Configuração dos Thresholds:")
        print(f"   • Threshold Absolute: {self.threshold_mutual_info:.6f}")
        print(f"   • Threshold Percentil {self.threshold_percentile}%: {stats['threshold_percentile_value']:.6f}")
        
        print(f"\n✅ Features Selecionadas:")
        print(f"   • Por Threshold Absolute: {stats['selected_by_threshold']} features")
        print(f"   • Por Percentil: {stats['selected_by_percentile']} features")
        print(f"   • Critério Combinado: {stats['selected_combined']} features")
        
        print(f"\n🏆 Top 10 Features:")
        top_10 = self.resultados['df_analysis'].head(10)
        for i, row in top_10.iterrows():
            status = "✅" if row['selected_combined'] else "❌"
            print(f"   {row['rank']:2d}. {status} {row['feature']}: {row['mutual_info_score']:.6f}")
        
        print("=" * 70)
    
    def get_feature_ranking(self) -> pd.DataFrame:
        """
        Retorna ranking completo das features.
        
        Returns:
            DataFrame: Ranking das features com scores e seleções
        """
        if self.resultados is None:
            raise ValueError("Execute calcular_mutual_information() primeiro.")
        
        return self.resultados['df_analysis'].copy()
    
    def atualizar_thresholds(self, 
                           threshold_mutual_info: float = None,
                           threshold_percentile: float = None):
        """
        Atualiza thresholds e recalcula seleções.
        
        Args:
            threshold_mutual_info (float): Novo threshold absoluto
            threshold_percentile (float): Novo threshold percentil
        """
        if self.resultados is None:
            raise ValueError("Execute calcular_mutual_information() primeiro.")
        
        if threshold_mutual_info is not None:
            self.threshold_mutual_info = threshold_mutual_info
            
        if threshold_percentile is not None:
            self.threshold_percentile = threshold_percentile
        
        # Recalcular seleções com novos thresholds
        df_analysis = self.resultados['df_analysis']
        threshold_value = np.percentile(df_analysis['mutual_info_score'], self.threshold_percentile)
        
        df_analysis['selected_by_percentile'] = df_analysis['mutual_info_score'] >= threshold_value
        df_analysis['selected_by_threshold'] = df_analysis['mutual_info_score'] >= self.threshold_mutual_info
        df_analysis['selected_combined'] = df_analysis['selected_by_percentile'] & df_analysis['selected_by_threshold']
        
        # Atualizar estatísticas
        self.resultados['statistics'].update({
            'threshold_percentile_value': threshold_value,
            'threshold_mutual_info_value': self.threshold_mutual_info,
            'selected_by_percentile': df_analysis['selected_by_percentile'].sum(),
            'selected_by_threshold': df_analysis['selected_by_threshold'].sum(),
            'selected_combined': df_analysis['selected_combined'].sum()
        })
        
        # Atualizar features selecionadas
        self.features_selecionadas = df_analysis[df_analysis['selected_combined']]['feature'].tolist()
        
        print(f"🔄 Thresholds atualizados:")
        print(f"   • MI Threshold: {self.threshold_mutual_info:.6f}")
        print(f"   • Percentil {self.threshold_percentile}%: {threshold_value:.6f}")
        print(f"   • Features selecionadas: {len(self.features_selecionadas)}")
    
    def analisar_features_por_ranges(self) -> Dict:
        """
        Analisa e classifica features baseado em ranges de importância predefinidos.
        
        Returns:
            Dict: Análise detalhada por ranges de importância
        """
        if self.resultados is None:
            raise ValueError("Execute calcular_mutual_information() primeiro.")
        
        # Definir ranges de importância
        ranges = [
            (0.0, 0.001, "Muito Baixa"),
            (0.001, 0.01, "Baixa"),
            (0.01, 0.05, "Moderada"),
            (0.05, 0.1, "Alta"),
            (0.1, float('inf'), "Muito Alta")
        ]
        
        df_analysis = self.resultados['df_analysis'].copy()
        
        # Aplicar classificação por ranges
        def classificar_importancia(score):
            for min_val, max_val, categoria in ranges:
                if min_val <= score < max_val:
                    return categoria
            return "Muito Alta"  # Para scores >= 0.1
        
        df_analysis['categoria_importancia'] = df_analysis['mutual_info_score'].apply(classificar_importancia)
        
        # Análise estatística por categoria
        analise_ranges = {}
        
        for min_val, max_val, categoria in ranges:
            features_categoria = df_analysis[df_analysis['categoria_importancia'] == categoria]
            
            if len(features_categoria) > 0:
                analise_ranges[categoria] = {
                    'range_min': min_val,
                    'range_max': max_val if max_val != float('inf') else 'inf',
                    'count': len(features_categoria),
                    'percentage': len(features_categoria) / len(df_analysis) * 100,
                    'features': features_categoria['feature'].tolist(),
                    'scores_min': features_categoria['mutual_info_score'].min(),
                    'scores_max': features_categoria['mutual_info_score'].max(),
                    'scores_mean': features_categoria['mutual_info_score'].mean(),
                    'scores_std': features_categoria['mutual_info_score'].std()
                }
            else:
                analise_ranges[categoria] = {
                    'range_min': min_val,
                    'range_max': max_val if max_val != float('inf') else 'inf',
                    'count': 0,
                    'percentage': 0.0,
                    'features': [],
                    'scores_min': None,
                    'scores_max': None,
                    'scores_mean': None,
                    'scores_std': None
                }
        
        # Armazenar resultado
        resultado_completo = {
            'analise_por_ranges': analise_ranges,
            'df_features_classificadas': df_analysis,
            'ranges_definicao': ranges,
            'resumo': {
                'total_features': len(df_analysis),
                'distribuicao_por_categoria': {cat: analise_ranges[cat]['count'] for cat in analise_ranges}
            }
        }
        
        return resultado_completo
    
    def imprimir_analise_ranges(self):
        """Imprime relatório detalhado da análise por ranges de importância."""
        analise = self.analisar_features_por_ranges()
        
        print(f"\n🎯 ANÁLISE DE FEATURES POR RANGES DE IMPORTÂNCIA")
        print("=" * 70)
        
        print(f"📊 Definição dos Ranges:")
        for min_val, max_val, categoria in analise['ranges_definicao']:
            max_str = f"{max_val}" if max_val != float('inf') else "∞"
            print(f"   • {categoria}: [{min_val} - {max_str})")
        
        print(f"\n📈 Distribuição das Features por Categoria:")
        total = analise['resumo']['total_features']
        
        for categoria, dados in analise['analise_por_ranges'].items():
            count = dados['count']
            percent = dados['percentage']
            
            if count > 0:
                print(f"\n🔶 {categoria}:")
                print(f"   • Quantidade: {count} features ({percent:.1f}%)")
                print(f"   • Range: [{dados['range_min']} - {dados['range_max']})")
                print(f"   • Scores: {dados['scores_min']:.6f} - {dados['scores_max']:.6f}")
                print(f"   • Média: {dados['scores_mean']:.6f} (±{dados['scores_std']:.6f})")
                
                # Mostrar algumas features exemplo
                if len(dados['features']) <= 5:
                    print(f"   • Features: {', '.join(dados['features'])}")
                else:
                    print(f"   • Exemplos: {', '.join(dados['features'][:5])} ...")
            else:
                print(f"\n🔸 {categoria}: Nenhuma feature encontrada")
        
        print(f"\n📊 Resumo Estatístico:")
        print(f"   • Total de features analisadas: {total}")
        
        # Mostrar features por categoria de alta importância
        features_importantes = []
        for categoria in ['Muito Alta', 'Alta', 'Moderada']:
            if categoria in analise['analise_por_ranges'] and analise['analise_por_ranges'][categoria]['count'] > 0:
                features_importantes.extend(analise['analise_por_ranges'][categoria]['features'])
        
        print(f"   • Features de importância Moderada ou superior: {len(features_importantes)}")
        print(f"   • Recomendação: Focar nas features de categoria 'Alta' e 'Muito Alta'")
        
        print("=" * 70)
    
    def plotar_analise_ranges(self, figsize: Tuple[int, int] = (14, 10)):
        """
        Plota visualizações da análise por ranges de importância.
        
        Args:
            figsize (tuple): Tamanho da figura
        """
        analise = self.analisar_features_por_ranges()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Gráfico de barras - Distribuição por categoria
        categorias = list(analise['analise_por_ranges'].keys())
        counts = [analise['analise_por_ranges'][cat]['count'] for cat in categorias]
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        
        bars = ax1.bar(categorias, counts, color=colors)
        ax1.set_title('Distribuição de Features por Categoria de Importância')
        ax1.set_ylabel('Número de Features')
        ax1.tick_params(axis='x', rotation=45)
        
        # Adicionar valores nas barras
        for bar, count in zip(bars, counts):
            if count > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        str(count), ha='center', va='bottom')
        
        # 2. Gráfico de pizza - Percentuais
        non_zero_cats = [cat for cat in categorias if analise['analise_por_ranges'][cat]['count'] > 0]
        non_zero_counts = [analise['analise_por_ranges'][cat]['count'] for cat in non_zero_cats]
        non_zero_colors = [colors[i] for i, cat in enumerate(categorias) if cat in non_zero_cats]
        
        if non_zero_counts:
            ax2.pie(non_zero_counts, labels=non_zero_cats, colors=non_zero_colors, autopct='%1.1f%%')
            ax2.set_title('Distribuição Percentual por Categoria')
        
        # 3. Histograma dos scores com ranges coloridos
        df_features = analise['df_features_classificadas']
        
        # Definir cores para cada categoria
        color_map = {'Muito Baixa': 'red', 'Baixa': 'orange', 'Moderada': 'yellow', 
                    'Alta': 'lightgreen', 'Muito Alta': 'green'}
        
        for categoria in categorias:
            features_cat = df_features[df_features['categoria_importancia'] == categoria]
            if len(features_cat) > 0:
                ax3.hist(features_cat['mutual_info_score'], bins=20, alpha=0.7, 
                        label=categoria, color=color_map.get(categoria, 'gray'))
        
        ax3.set_xlabel('Mutual Information Score')
        ax3.set_ylabel('Frequência')
        ax3.set_title('Distribuição dos Scores por Categoria')
        ax3.legend()
        ax3.grid(alpha=0.3)
        
        # 4. Boxplot por categoria
        box_data = []
        box_labels = []
        
        for categoria in categorias:
            features_cat = df_features[df_features['categoria_importancia'] == categoria]
            if len(features_cat) > 0:
                box_data.append(features_cat['mutual_info_score'])
                box_labels.append(f"{categoria}\n(n={len(features_cat)})")
        
        if box_data:
            bp = ax4.boxplot(box_data, labels=box_labels, patch_artist=True)
            
            # Colorir os boxplots
            for i, patch in enumerate(bp['boxes']):
                categoria = categorias[i] if i < len(categorias) else categorias[0]
                patch.set_facecolor(color_map.get(categoria, 'gray'))
        
        ax4.set_ylabel('Mutual Information Score')
        ax4.set_title('Distribuição dos Scores por Categoria (Boxplot)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def selecionar_features_por_categoria(self, categorias_desejadas: List[str]) -> List[str]:
        """
        Seleciona features baseado nas categorias de importância.
        
        Args:
            categorias_desejadas (List[str]): Lista de categorias a selecionar
            
        Returns:
            List[str]: Features das categorias selecionadas
        """
        analise = self.analisar_features_por_ranges()
        df_features = analise['df_features_classificadas']
        
        features_selecionadas = []
        
        for categoria in categorias_desejadas:
            if categoria in analise['analise_por_ranges']:
                features_cat = df_features[df_features['categoria_importancia'] == categoria]['feature'].tolist()
                features_selecionadas.extend(features_cat)
        
        print(f"🎯 Features selecionadas das categorias {categorias_desejadas}:")
        print(f"   • Total: {len(features_selecionadas)} features")
        
        for categoria in categorias_desejadas:
            count = len(df_features[df_features['categoria_importancia'] == categoria])
            print(f"   • {categoria}: {count} features")
        
        return features_selecionadas


def main():
    """
    Função de exemplo demonstrando o uso da classe FeatureAnalyzer.
    """
    print("🧠 Demonstração da classe FeatureAnalyzer")
    print("=" * 60)
    
    try:
        # Criar dados sintéticos para demonstração
        from sklearn.datasets import make_classification, make_regression
        
        print("📊 Criando dados sintéticos para demonstração...")
        
        # Dados de classificação
        X_class, y_class = make_classification(n_samples=1000, n_features=20, 
                                             n_informative=10, n_redundant=5, 
                                             random_state=42)
        
        # Converter para DataFrame
        feature_names = [f'feature_{i+1:02d}' for i in range(X_class.shape[1])]
        df_X = pd.DataFrame(X_class, columns=feature_names)
        
        print(f"   • Dataset criado: {df_X.shape}")
        print(f"   • Classes únicas: {len(np.unique(y_class))}")
        
        # Criar analisador
        analyzer = FeatureAnalyzer(
            threshold_mutual_info=0.05,
            threshold_percentile=70.0,
            random_state=42
        )
        
        # Executar análise
        resultados = analyzer.calcular_mutual_information(df_X, y_class)
        
        # Imprimir relatório
        analyzer.imprimir_relatorio()
        
        # Testar diferentes seleções
        print(f"\n🎯 Testando diferentes métodos de seleção:")
        
        top_10 = analyzer.selecionar_top_features(10)
        by_threshold = analyzer.selecionar_features_por_threshold(metodo='absolute')
        by_percentile = analyzer.selecionar_features_por_threshold(metodo='percentile')
        
        # Testar nova funcionalidade de análise por ranges
        print(f"\n🎯 Testando análise por ranges de importância:")
        analyzer.imprimir_analise_ranges()
        
        # Testar seleção por categoria
        features_importantes = analyzer.selecionar_features_por_categoria(['Alta', 'Muito Alta'])
        features_moderadas = analyzer.selecionar_features_por_categoria(['Moderada', 'Alta', 'Muito Alta'])
        
        print(f"\n✅ FeatureAnalyzer com análise por ranges pronto para uso!")
        
    except Exception as e:
        print(f"⚠️ Exemplo não executado completamente: {e}")
        print("   Instale scikit-learn para testar completamente.")


if __name__ == "__main__":
    main()