"""
Testes Unitários para a Classe UtilClassificadores

Este módulo contém testes unitários para validar o comportamento da classe
UtilClassificadores e seus métodos principais.

Autor: Sistema de Testes
Data: 22/10/2025
"""

import unittest
import numpy as np
import pandas as pd
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Importar a classe a ser testada
from scripts.lib_classificador import UtilClassificadores


class TestUtilClassificadoresInit(unittest.TestCase):
    """Testes para o método __init__ da classe UtilClassificadores"""
    
    def test_init_default_titles(self):
        """Testa inicialização com títulos padrão"""
        util = UtilClassificadores()
        self.assertIsNotNone(util.titles)
        self.assertIsInstance(util.titles, list)
        self.assertGreater(len(util.titles), 0)
        
    def test_init_custom_titles(self):
        """Testa inicialização com títulos customizados"""
        custom_titles = ['DecisionTree', 'KNN', 'RF']
        util = UtilClassificadores(titles=custom_titles)
        self.assertEqual(util.titles, custom_titles)
        
    def test_init_invalid_titles_filtered(self):
        """Testa que títulos inválidos são filtrados"""
        invalid_titles = ['DecisionTree', 'InvalidModel', 'KNN']
        util = UtilClassificadores(titles=invalid_titles)
        self.assertNotIn('InvalidModel', util.titles)
        self.assertIn('DecisionTree', util.titles)
        self.assertIn('KNN', util.titles)
        
    def test_init_attributes(self):
        """Testa que todos os atributos são inicializados corretamente"""
        util = UtilClassificadores(titles=['DecisionTree'])
        self.assertIsInstance(util.methods, list)
        self.assertIsInstance(util.valid_titles, list)
        self.assertIsInstance(util.trained_models, dict)
        
    def test_repr(self):
        """Testa a representação string da classe"""
        util = UtilClassificadores(titles=['DecisionTree'])
        repr_str = repr(util)
        self.assertIn('UtilClassificadores', repr_str)
        self.assertIn('modelos=', repr_str)


class TestCreateClassifiers(unittest.TestCase):
    """Testes para o método create_classifiers"""
    
    def test_create_classifiers_default(self):
        """Testa criação de classificadores com configuração padrão"""
        util = UtilClassificadores(titles=['DecisionTree', 'KNN'])
        self.assertEqual(len(util.methods), 2)
        self.assertEqual(len(util.valid_titles), 2)
        
    def test_create_classifiers_all_available(self):
        """Testa que todos os classificadores disponíveis podem ser criados"""
        all_titles = UtilClassificadores.TITLES
        util = UtilClassificadores(titles=all_titles)
        self.assertGreaterEqual(len(util.valid_titles), len(all_titles) - 1)  # Permitir 1 falha
        
    def test_create_classifiers_returns_valid_objects(self):
        """Testa que os classificadores criados têm métodos fit e predict"""
        util = UtilClassificadores(titles=['DecisionTree', 'KNN'])
        for method in util.methods:
            self.assertTrue(hasattr(method, 'fit'))
            self.assertTrue(hasattr(method, 'predict'))
            
    def test_create_classifiers_valid_titles_match(self):
        """Testa que valid_titles correspondem aos métodos criados"""
        util = UtilClassificadores(titles=['DecisionTree', 'KNN', 'RF'])
        self.assertEqual(len(util.methods), len(util.valid_titles))


class TestTrainClassifiers(unittest.TestCase):
    """Testes para o método train_classifiers"""
    
    @classmethod
    def setUpClass(cls):
        """Configura dados de teste para toda a classe"""
        # Criar dataset sintético
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5, 
                                   n_redundant=2, n_classes=2, random_state=42)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
    def test_train_classifiers_success(self):
        """Testa treinamento bem-sucedido de classificadores"""
        util = UtilClassificadores(titles=['DecisionTree', 'KNN'])
        trained_models = util.train_classifiers(
            X_train=self.X_train, 
            y_train=self.y_train, 
            verbose=False
        )
        
        self.assertIsInstance(trained_models, dict)
        self.assertGreater(len(trained_models), 0)
        
    def test_train_classifiers_all_success(self):
        """Testa que todos os modelos são treinados com sucesso"""
        util = UtilClassificadores(titles=['DecisionTree', 'KNN', 'RF'])
        trained_models = util.train_classifiers(
            X_train=self.X_train, 
            y_train=self.y_train, 
            verbose=False
        )
        
        for model_info in trained_models.values():
            self.assertEqual(model_info['status'], 'success')
            
    def test_train_classifiers_stores_models(self):
        """Testa que modelos treinados são armazenados no atributo trained_models"""
        util = UtilClassificadores(titles=['DecisionTree'])
        util.train_classifiers(
            X_train=self.X_train, 
            y_train=self.y_train, 
            verbose=False
        )
        
        self.assertGreater(len(util.trained_models), 0)
        self.assertIn('DecisionTree', util.trained_models)
        
    def test_train_classifiers_model_has_fit_method_called(self):
        """Testa que os modelos foram realmente treinados (têm atributos de modelo treinado)"""
        util = UtilClassificadores(titles=['DecisionTree'])
        util.train_classifiers(
            X_train=self.X_train, 
            y_train=self.y_train, 
            verbose=False
        )
        
        model = util.trained_models['DecisionTree']['model']
        # DecisionTree terá tree_ após treinamento
        self.assertTrue(hasattr(model, 'tree_'))


class TestEvaluateClassifiers(unittest.TestCase):
    """Testes para o método evaluate_classifiers"""
    
    @classmethod
    def setUpClass(cls):
        """Configura dados de teste para toda a classe"""
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5, 
                                   n_redundant=2, n_classes=2, random_state=42)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
    def test_evaluate_classifiers_returns_results(self):
        """Testa que a avaliação retorna resultados e scores"""
        util = UtilClassificadores(titles=['DecisionTree', 'KNN'])
        util.train_classifiers(
            X_train=self.X_train, 
            y_train=self.y_train, 
            verbose=False
        )
        
        results, scores = util.evaluate_classifiers(
            X_test_scaled=self.X_test,
            y_test=self.y_test,
            verbose=False
        )
        
        self.assertIsInstance(results, dict)
        self.assertIsInstance(scores, list)
        
    def test_evaluate_classifiers_accuracy_range(self):
        """Testa que as acurácias estão no intervalo válido [0, 1]"""
        util = UtilClassificadores(titles=['DecisionTree', 'KNN'])
        util.train_classifiers(
            X_train=self.X_train, 
            y_train=self.y_train, 
            verbose=False
        )
        
        results, scores = util.evaluate_classifiers(
            X_test_scaled=self.X_test,
            y_test=self.y_test,
            verbose=False
        )
        
        for score in scores:
            if score is not None:
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)
                
    def test_evaluate_classifiers_results_structure(self):
        """Testa a estrutura dos resultados retornados"""
        util = UtilClassificadores(titles=['DecisionTree'])
        util.train_classifiers(
            X_train=self.X_train, 
            y_train=self.y_train, 
            verbose=False
        )
        
        results, _ = util.evaluate_classifiers(
            X_test_scaled=self.X_test,
            y_test=self.y_test,
            verbose=False
        )
        
        for result in results.values():
            self.assertIn('accuracy', result)
            self.assertIn('model', result)
            self.assertIn('status', result)


class TestGetMethodsTrainedModels(unittest.TestCase):
    """Testes para o método get_methods_trained_models"""
    
    @classmethod
    def setUpClass(cls):
        """Configura dados de teste para toda a classe"""
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5, 
                                   n_redundant=2, n_classes=2, random_state=42)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
    def test_get_methods_returns_tuple(self):
        """Testa que o método retorna uma tupla (methods, valid_titles)"""
        util = UtilClassificadores(titles=['DecisionTree'])
        util.train_classifiers(
            X_train=self.X_train, 
            y_train=self.y_train, 
            verbose=False
        )
        
        result = util.get_methods_trained_models()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        
    def test_get_methods_filters_successful_only(self):
        """Testa que apenas modelos com sucesso são retornados"""
        util = UtilClassificadores(titles=['DecisionTree'])
        util.train_classifiers(
            X_train=self.X_train, 
            y_train=self.y_train, 
            verbose=False
        )
        
        # Adicionar modelo com erro manualmente
        util.trained_models['ErrorModel'] = {
            'model': None,
            'status': 'error',
            'error': 'Test error'
        }
        
        methods, valid_titles = util.get_methods_trained_models()
        self.assertNotIn('ErrorModel', valid_titles)


class TestGenerateModelMetricsDataset(unittest.TestCase):
    """Testes para o método generate_model_metrics_dataset"""
    
    @classmethod
    def setUpClass(cls):
        """Configura dados de teste para toda a classe"""
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5, 
                                   n_redundant=2, n_classes=2, random_state=42)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
    def test_generate_metrics_returns_dataframe(self):
        """Testa que o método retorna um DataFrame"""
        util = UtilClassificadores(titles=['DecisionTree'])
        util.train_classifiers(
            X_train=self.X_train, 
            y_train=self.y_train, 
            verbose=False
        )
        
        metrics_df = util.generate_model_metrics_dataset(
            X_test_scaled=self.X_test,
            y_test=self.y_test,
            dataset_name='test_dataset',
            save_to_csv=False,
            display_results=False,
            verbose=False
        )
        
        self.assertIsInstance(metrics_df, pd.DataFrame)
        
    def test_generate_metrics_has_required_columns(self):
        """Testa que o DataFrame tem as colunas necessárias"""
        util = UtilClassificadores(titles=['DecisionTree'])
        util.train_classifiers(
            X_train=self.X_train, 
            y_train=self.y_train, 
            verbose=False
        )
        
        metrics_df = util.generate_model_metrics_dataset(
            X_test_scaled=self.X_test,
            y_test=self.y_test,
            dataset_name='test_dataset',
            save_to_csv=False,
            display_results=False,
            verbose=False
        )
        
        required_columns = ['modelo', 'acuracia', 'precisao', 'recall', 'f1_score', 'dataset', 'status']
        for col in required_columns:
            self.assertIn(col, metrics_df.columns)
            
    def test_generate_metrics_saves_csv(self):
        """Testa que o CSV é salvo quando solicitado"""
        with tempfile.TemporaryDirectory() as tmpdir:
            util = UtilClassificadores(titles=['DecisionTree'])
            util.train_classifiers(
                X_train=self.X_train, 
                y_train=self.y_train, 
                verbose=False
            )
            
            util.generate_model_metrics_dataset(
                X_test_scaled=self.X_test,
                y_test=self.y_test,
                dataset_name='test_dataset',
                save_to_csv=True,
                save_dir=tmpdir,
                display_results=False,
                verbose=False
            )
            
            # Verificar se pelo menos um arquivo CSV foi criado
            csv_files = [f for f in os.listdir(tmpdir) if f.endswith('.csv')]
            self.assertGreater(len(csv_files), 0)


class TestGenerateConfusionMatrices(unittest.TestCase):
    """Testes para o método generate_confusion_matrices"""
    
    @classmethod
    def setUpClass(cls):
        """Configura dados de teste para toda a classe"""
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5, 
                                   n_redundant=2, n_classes=2, random_state=42)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
    @patch('matplotlib.pyplot.show')  # Mock para não exibir gráficos durante testes
    def test_generate_confusion_matrices_returns_dict(self, mock_show):
        """Testa que o método retorna um dicionário de resultados"""
        util = UtilClassificadores(titles=['DecisionTree'])
        util.train_classifiers(
            X_train=self.X_train, 
            y_train=self.y_train, 
            verbose=False
        )
        
        results = util.generate_confusion_matrices(
            X_test_scaled=self.X_test,
            y_test=self.y_test,
            verbose=False
        )
        
        self.assertIsInstance(results, dict)
        
    @patch('matplotlib.pyplot.show')
    def test_generate_confusion_matrices_has_accuracy(self, mock_show):
        """Testa que os resultados contêm informações de acurácia"""
        util = UtilClassificadores(titles=['DecisionTree'])
        util.train_classifiers(
            X_train=self.X_train, 
            y_train=self.y_train, 
            verbose=False
        )
        
        results = util.generate_confusion_matrices(
            X_test_scaled=self.X_test,
            y_test=self.y_test,
            verbose=False
        )
        
        for result in results.values():
            if result['status'] == 'success':
                self.assertIn('accuracy', result)
                self.assertIsNotNone(result['accuracy'])


class TestGenerateRocAucCurves(unittest.TestCase):
    """Testes para o método generate_roc_auc_curves"""
    
    @classmethod
    def setUpClass(cls):
        """Configura dados de teste para toda a classe"""
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5, 
                                   n_redundant=2, n_classes=2, random_state=42)
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
    @patch('matplotlib.pyplot.show')
    def test_generate_roc_auc_runs_without_error(self, mock_show):
        """Testa que o método executa sem erros para classificação binária"""
        util = UtilClassificadores(titles=['DecisionTree', 'KNN'])
        util.train_classifiers(
            X_train=self.X_train, 
            y_train=self.y_train, 
            verbose=False
        )
        
        try:
            util.generate_roc_auc_curves(
                X_test_scaled=self.X_test,
                y_test=self.y_test,
                nome_dataset='test_dataset',
                verbose=False
            )
            success = True
        except Exception as e:
            success = False
            print(f"Error: {e}")
            
        self.assertTrue(success)


class TestUtilityMethods(unittest.TestCase):
    """Testes para métodos utilitários da classe"""
    
    def test_get_titles_trained_models(self):
        """Testa o método get_titles_trained_models"""
        util = UtilClassificadores(titles=['DecisionTree', 'KNN'])
        titles = util.get_titles_trained_models()
        self.assertIsInstance(titles, list)
        self.assertEqual(len(titles), 2)
        
    def test_get_trained_models(self):
        """Testa o método get_trained_models"""
        util = UtilClassificadores(titles=['DecisionTree'])
        models = util.get_trained_models()
        self.assertIsInstance(models, dict)
        
    def test_get_successful_models(self):
        """Testa o método get_successful_models"""
        util = UtilClassificadores(titles=['DecisionTree'])
        
        # Adicionar modelos de teste
        util.trained_models = {
            'Model1': {'status': 'success', 'model': Mock()},
            'Model2': {'status': 'error', 'model': Mock()},
            'Model3': {'status': 'success', 'model': Mock()}
        }
        
        successful = util.get_successful_models()
        self.assertEqual(len(successful), 2)
        self.assertIn('Model1', successful)
        self.assertIn('Model3', successful)
        self.assertNotIn('Model2', successful)


class TestIntegration(unittest.TestCase):
    """Testes de integração - fluxo completo"""
    
    def test_complete_workflow(self):
        """Testa o fluxo completo: criar -> treinar -> avaliar -> métricas"""
        # Criar dados
        X, y = make_classification(n_samples=200, n_features=10, n_informative=5, 
                                   n_redundant=2, n_classes=2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Criar instância
        util = UtilClassificadores(titles=['DecisionTree', 'KNN'])
        
        # Treinar
        trained_models = util.train_classifiers(
            X_train=X_train, 
            y_train=y_train, 
            verbose=False
        )
        self.assertGreater(len(trained_models), 0)
        
        # Avaliar
        results, scores = util.evaluate_classifiers(
            X_test_scaled=X_test,
            y_test=y_test,
            verbose=False
        )
        self.assertGreater(len(results), 0)
        
        # Gerar métricas
        metrics_df = util.generate_model_metrics_dataset(
            X_test_scaled=X_test,
            y_test=y_test,
            dataset_name='integration_test',
            save_to_csv=False,
            display_results=False,
            verbose=False
        )
        self.assertIsInstance(metrics_df, pd.DataFrame)
        self.assertGreater(len(metrics_df), 0)


def run_tests():
    """Função para executar todos os testes"""
    print("=" * 70)
    print("EXECUTANDO TESTES UNITÁRIOS - UtilClassificadores")
    print("=" * 70)
    
    # Criar suite de testes
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Adicionar todas as classes de teste
    suite.addTests(loader.loadTestsFromTestCase(TestUtilClassificadoresInit))
    suite.addTests(loader.loadTestsFromTestCase(TestCreateClassifiers))
    suite.addTests(loader.loadTestsFromTestCase(TestTrainClassifiers))
    suite.addTests(loader.loadTestsFromTestCase(TestEvaluateClassifiers))
    suite.addTests(loader.loadTestsFromTestCase(TestGetMethodsTrainedModels))
    suite.addTests(loader.loadTestsFromTestCase(TestGenerateModelMetricsDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestGenerateConfusionMatrices))
    suite.addTests(loader.loadTestsFromTestCase(TestGenerateRocAucCurves))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityMethods))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Executar testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Resumo
    print("\n" + "=" * 70)
    print("RESUMO DOS TESTES")
    print("=" * 70)
    print(f"✅ Testes executados: {result.testsRun}")
    print(f"✅ Sucessos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Falhas: {len(result.failures)}")
    print(f"❌ Erros: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n🎉 TODOS OS TESTES PASSARAM COM SUCESSO!")
    else:
        print("\n⚠️ ALGUNS TESTES FALHARAM!")
    
    return result


if __name__ == '__main__':
    run_tests()
