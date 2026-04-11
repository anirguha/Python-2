import os
os.environ['JAVA_HOME'] = '/Library/Java/JavaVirtualMachines/amazon-corretto-11.jdk/Contents/Home'
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("PySparkSanityCheck").getOrCreate()
df = spark.range(5)
df.show()
spark.stop()