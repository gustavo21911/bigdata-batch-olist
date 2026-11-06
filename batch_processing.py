from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count

# Crear la sesión de Spark para procesamiento Batch
spark = SparkSession.builder \
    .appName("OlistBatchAnalysis") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=== 1. CARGA DE DATOS ===")
# Cargar el archivo CSV con cabecera e inferencia automática de esquemas
df = spark.read.option("header", "true").option("inferSchema", "true").csv("olist_orders_dataset.csv")

print("Esquema del Dataset:")
df.printSchema()

print("Primeros registros cargados:")
df.show(5, truncate=False)

print("=== 2. ANÁLISIS EXPLORATORIO (EDA) Y TRANSFORMACIONES ===")
# Total de registros en el lote
total_records = df.count()
print(f"Total de registros analizados: {total_records}")

# Agrupar y contar por estado de la orden (order_status)
print("Agrupación por estado de la orden:")
status_counts = df.groupBy("order_status").agg(count("order_id").alias("total_orders"))
status_counts.show()

# Filtrar únicamente las órdenes que se encuentran en estado 'delivered'
print("=== 3. FILTRADO Y VISUALIZACIÓN DE RESULTADOS ===")
delivered_df = df.filter(col("order_status") == "delivered")
print("Órdenes con estado 'delivered':")
delivered_df.show(truncate=False)

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from pyspark.sql.functions import col, count, date_format

print("=== 4. ANÁLISIS AVANZADO Y GRÁFICO TEMPORAL ===")

# Extraer el año y mes de la fecha de compra para ver la tendencia temporal
df_temporal = (
    df.withColumn(
        "mes_compra", date_format(col("order_purchase_timestamp"), "yyyy-MM")
    )
    .groupBy("mes_compra")
    .agg(count("order_id").alias("total_ordenes"))
    .orderBy("mes_compra")
)

# Mostrar los datos resultantes en consola
print("Tendencia de órdenes por mes:")
df_temporal.show(20, truncate=False)

# Convertir a Pandas para generar un gráfico de líneas y puntos más completo
pdf = df_temporal.toPandas()

# Configurar el tamaño de la figura
plt.figure(figsize=(12, 6))

# Crear un gráfico de líneas con marcadores
plt.plot(
    pdf["mes_compra"],
    pdf["total_ordenes"],
    marker="o",
    linestyle="-",
    color="#1f77b4",
    linewidth=2.5,
    markersize=6,
)

# Personalización formal para el informe
plt.xlabel("Período de Compra (Año - Mes)", fontsize=11)
plt.ylabel("Cantidad de Pedidos", fontsize=11)
plt.title(
    "Evolución Temporal del Volumen de Pedidos - Dataset Olist",
    fontsize=13,
    weight="bold",
)
plt.xticks(rotation=45)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

# Guardar la nueva gráfica en alta resolución
plt.savefig("grafico_olist.png", dpi=300)
print(
    "¡Gráfico avanzado actualizado y guardado exitosamente como"
    " 'grafico_olist.png'!"
)



# Detener la sesión de Spark
spark.stop()
