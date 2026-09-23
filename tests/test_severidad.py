"""Regresiones del baseline de pérdida, sin descargar datos de Kaggle."""
import unittest

import numpy as np
import pandas as pd

from src.features import calcular_perdida, estimar_severidad


class SeveridadTest(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({
            "loan_status": ["Charged Off", "Charged Off", "Fully Paid"],
            "default": [1, 1, 0],
            "funded_amnt": [100.0, 100.0, 100.0],
            "total_rec_prncp": [80.0, 20.0, 100.0],
            "recoveries": [0.0, 60.0, 0.0],
            "collection_recovery_fee": [0.0, 0.0, 0.0],
        })
        self.train = self.train.join(calcular_perdida(self.train))

    def test_media_del_producto_con_severidad_y_exposicion_dependientes(self):
        # Ambos defaults pierden 20/100. Multiplicar las medias de LGD y
        # EAD/monto daría 0.3125 y sobreestimaría esta pérdida condicional.
        defaults = self.train[self.train["default"] == 1]
        producto_medias = defaults["lgd"].mean() * (
            defaults["ead"] / defaults["funded_amnt"]
        ).mean()
        self.assertAlmostEqual(producto_medias, 0.3125)
        self.assertAlmostEqual(estimar_severidad(self.train), 0.2)
        self.assertTrue(pd.isna(self.train.loc[2, "perdida_sobre_monto"]))

    def test_no_promedia_pagados_ni_modifica_datos(self):
        original = self.train.copy(deep=True)
        pagados = self.train.loc[[2]].copy()
        muchos_pagados = pd.concat([self.train, pagados, pagados], ignore_index=True)
        self.assertAlmostEqual(estimar_severidad(muchos_pagados), 0.2)
        pd.testing.assert_frame_equal(self.train, original)

    def test_no_estima_sin_defaults(self):
        with self.assertRaisesRegex(ValueError, "al menos un default"):
            estimar_severidad(self.train[self.train["default"] == 0])

    def test_no_omite_perdidas_invalidas_de_defaults(self):
        for valor in [np.nan, np.inf, -0.1, 1.1]:
            with self.subTest(valor=valor):
                train = self.train.copy()
                train.loc[0, "perdida_sobre_monto"] = valor
                with self.assertRaisesRegex(ValueError, "pérdida finita"):
                    estimar_severidad(train)

    def test_no_omite_etiquetas_desconocidas(self):
        train = self.train.astype({"default": float})
        train.loc[0, "default"] = np.nan
        with self.assertRaisesRegex(ValueError, "default debe"):
            estimar_severidad(train)


if __name__ == "__main__":
    unittest.main()
