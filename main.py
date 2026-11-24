import joblib
import pandas as pd
from apify import Actor

MODEL_PATH = "modelo_gas_extratrees.pkl"
CSV_KEY = "matriz_resultados.csv"


async def main() -> None:
    async with Actor:
        modelo = joblib.load(MODEL_PATH)
        FEATURES = list(modelo.feature_names_in_)
        Actor.log.info(f"Modelo cargado. {len(FEATURES)} features. Clases: {list(modelo.classes_)}")

        actor_input = await Actor.get_input() or {}
        lecturas = actor_input.get("lecturas", [])
        if not lecturas:
            raise ValueError("El input no trae 'lecturas'.")

        df = pd.DataFrame(lecturas)
        faltantes = [c for c in FEATURES if c not in df.columns]
        if faltantes:
            raise ValueError(f"Faltan {len(faltantes)} columnas. Primeras: {faltantes[:5]}")

        X = df[FEATURES]
        df_out = df.copy()
        df_out["prediccion"] = modelo.predict(X)
        df_out["confianza"] = modelo.predict_proba(X).max(axis=1).round(4)

        await Actor.push_data(df_out.to_dict(orient="records"))
        await Actor.set_value(CSV_KEY, df_out.to_csv(index=False), content_type="text/csv")
        Actor.log.info(f"{CSV_KEY} escrito con {len(df_out)} filas.")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
