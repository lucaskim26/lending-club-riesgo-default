"""Casos sintéticos: no requieren descargar el dataset de Kaggle."""
import unittest

import numpy as np
import pandas as pd

from src.evaluacion import metricas_pd, tabla_por_banda


class MetricasPDTest(unittest.TestCase):
    def test_mismo_ranking_no_implica_mismas_probabilidades(self):
        y = [0, 0, 1, 1]
        buenas = metricas_pd(y, [0.1, 0.2, 0.8, 0.9])
        desplazadas = metricas_pd(y, [0.6, 0.7, 0.8, 0.9])
        for metrica in ("roc_auc", "average_precision", "ks"):
            self.assertEqual(buenas[metrica], 1.0)
            self.assertEqual(buenas[metrica], desplazadas[metrica])
        self.assertAlmostEqual(buenas["brier"], 0.025)
        self.assertAlmostEqual(buenas["log_loss"], -np.log([0.9, 0.8, 0.8, 0.9]).mean())
        self.assertLess(buenas["brier"], desplazadas["brier"])
        self.assertLess(buenas["log_loss"], desplazadas["log_loss"])

    def test_scores_empatados_y_average_precision_no_trapezoidal(self):
        resultado = metricas_pd([0, 0, 0, 1], [0.25] * 4)
        self.assertEqual(resultado["roc_auc"], 0.5)
        self.assertEqual(resultado["ks"], 0.0)
        self.assertEqual(resultado["average_precision"], 0.25)

    def test_ks_bilateral_no_indica_direccion_del_score(self):
        resultado = metricas_pd([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
        self.assertEqual(resultado["roc_auc"], 0.0)
        self.assertEqual(resultado["ks"], 1.0)

    def test_probabilidades_extremas_producen_log_loss_finita(self):
        resultado = metricas_pd([0, 1], [1, 0])
        self.assertTrue(np.isfinite(resultado["log_loss"]))
        self.assertEqual(resultado["brier"], 1.0)

    def test_rechaza_una_sola_clase(self):
        with self.assertRaisesRegex(ValueError, "pagados y en default"):
            metricas_pd([0, 0], [0.1, 0.2])


class BandasTest(unittest.TestCase):
    def test_limites_perdidas_y_prima_por_prestamo(self):
        tabla = tabla_por_banda(
            [0, 1, 0, 1, 1], [0, 0.1, 0.2, 0.5, 1], [0, 0.2, 0.5, 1],
            perdida_sobre_monto=[np.nan, 0.8, np.nan, 0.6, 1.0], severidad=0.5,
        )
        self.assertEqual(tabla["cantidad"].tolist(), [2, 1, 2])
        np.testing.assert_allclose(tabla["proporcion"], [0.4, 0.2, 0.4])
        np.testing.assert_allclose(tabla["pd_media"], [0.05, 0.2, 0.75])
        np.testing.assert_allclose(tabla["tasa_default_observada"], [0.5, 0, 1])
        # Incluye pagados en el denominador: la primera banda pierde 0.8 / 2.
        np.testing.assert_allclose(tabla["perdida_media_observada"], [0.4, 0, 0.8])
        np.testing.assert_allclose(tabla["prima_media"], [0.025, 0.1, 0.375])
        self.assertEqual(tabla["incluye_limite_superior"].tolist(), [False, False, True])

    def test_conserva_bandas_vacias_y_admite_muestra_sin_defaults(self):
        tabla = tabla_por_banda([0, 0], [0.1, 0.9], [0, 0.2, 0.8, 1])
        vacia = tabla.loc[1]
        self.assertEqual(vacia["cantidad"], 0)
        self.assertEqual(vacia["proporcion"], 0)
        self.assertTrue(np.isnan(vacia["pd_media"]))
        self.assertTrue(np.isnan(vacia["tasa_default_observada"]))
        self.assertNotIn("prima_media", tabla)
        self.assertNotIn("perdida_media_observada", tabla)

    def test_series_con_indices_coincidentes_conservan_posicion(self):
        indice = ["prestamo-b", "prestamo-a"]
        tabla = tabla_por_banda(
            pd.Series([1, 0], index=indice), pd.Series([0.1, 0.9], index=indice),
            [0, 0.5, 1], perdida_sobre_monto=pd.Series([0.6, np.nan], index=indice),
        )
        self.assertEqual(tabla["perdida_media_observada"].tolist(), [0.6, 0.0])

    def test_rechaza_cortes_incompletos_repetidos_o_desordenados(self):
        for cortes in ([0.1, 1], [0, 0.9], [0, 0.5, 0.5, 1], [0, 0.8, 0.2, 1],
                       [0, np.nan, 1], [0, np.inf, 1], [0], [[0, 1]]):
            with self.subTest(cortes=cortes), self.assertRaises(ValueError):
                tabla_por_banda([0, 1], [0.1, 0.9], cortes)

    def test_rechaza_perdidas_incompletas_o_incompatibles_con_target(self):
        for perdida in ([0, np.nan], [np.nan, np.inf], [0, -0.1], [0, 1.1], [0.1, 0.8]):
            with self.subTest(perdida=perdida), self.assertRaises(ValueError):
                tabla_por_banda([0, 1], [0.1, 0.9], [0, 1], perdida_sobre_monto=perdida)

    def test_rechaza_severidad_no_escalar_o_fuera_de_rango(self):
        for severidad in (np.nan, np.inf, -0.1, 1.1, [0.5], "invalida"):
            with self.subTest(severidad=severidad), self.assertRaises(ValueError):
                tabla_por_banda([0, 1], [0.1, 0.9], [0, 1], severidad=severidad)


class ValidacionTest(unittest.TestCase):
    def test_rechaza_vectores_invalidos_en_ambas_funciones(self):
        for y, prob in (([], []), ([0, 1], [0.1]), ([0, 2], [0.1, 0.9]),
                        ([0, np.nan], [0.1, 0.9]), ([0, 1], [0.1, np.nan]),
                        ([0, 1], [0.1, np.inf]), ([0, 1], [-0.1, 0.9]),
                        ([0, 1], [0.1, 1.1]), ([0, 1], [[0.1], [0.9]])):
            with self.subTest(y=y, prob=prob):
                with self.assertRaises(ValueError):
                    metricas_pd(y, prob)
                with self.assertRaises(ValueError):
                    tabla_por_banda(y, prob, [0, 1])

    def test_rechaza_series_desordenadas_aun_con_los_mismos_ids(self):
        y = pd.Series([0, 1], index=[10, 20])
        prob = pd.Series([0.9, 0.1], index=[20, 10])
        with self.assertRaisesRegex(ValueError, "índice"):
            metricas_pd(y, prob)
        with self.assertRaisesRegex(ValueError, "índice"):
            tabla_por_banda(y, prob, [0, 1])

    def test_rechaza_indice_de_perdida_incompatible_aun_si_y_es_lista(self):
        with self.assertRaisesRegex(ValueError, "índice"):
            tabla_por_banda(
                [0, 1], pd.Series([0.1, 0.9], index=[10, 20]), [0, 1],
                perdida_sobre_monto=pd.Series([0.8, np.nan], index=[20, 10]),
            )


if __name__ == "__main__":
    unittest.main()
