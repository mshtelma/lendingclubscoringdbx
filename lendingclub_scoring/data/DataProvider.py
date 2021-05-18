from pyspark.sql import Window
from pyspark.sql.functions import *
from pyspark.sql.types import *
from sklearn.model_selection import train_test_split


predictors = ["term", "home_ownership", "purpose", "addr_state", "verification_status", "application_type",
              "loan_amnt", "emp_length", "annual_inc", "dti", "delinq_2yrs", "revol_util", "total_acc",
              "credit_length_in_years", "int_rate", "net", "issue_year"]
target = 'bad_loan'

class LendingClubDataProvider():
    def __init__(self, spark, input_path, limit=None):
        self.spark = spark
        self.input_path = input_path
        self.limit = limit

    def load_and_transform_data(self):
        df = self.spark.read.format("parquet").load(self.input_path)
        if self.limit:
            df = df.limit(self.limit)

        df = df.select("loan_status", "int_rate", "revol_util", "issue_d", "earliest_cr_line", "emp_length",
                       "verification_status", "total_pymnt", "loan_amnt", "grade", "annual_inc", "dti", "addr_state",
                       "term", "home_ownership", "purpose", "application_type", "delinq_2yrs", "total_acc")

        df = df.filter(df.loan_status.isin(["Default", "Charged Off", "Fully Paid"])) \
            .withColumn("bad_loan", (~(df.loan_status == "Fully Paid")).cast("string"))

        df = df.withColumn('int_rate', regexp_replace('int_rate', '%', '').cast('float')) \
            .withColumn('revol_util', regexp_replace('revol_util', '%', '').cast('float')) \
            .withColumn('issue_year', substring(df.issue_d, 5, 4).cast('double')) \
            .withColumn('earliest_year', substring(df.earliest_cr_line, 5, 4).cast('double'))
        df = df.withColumn('credit_length_in_years', (df.issue_year - df.earliest_year))

        df = df.withColumn('emp_length', trim(regexp_replace(df.emp_length, "([ ]*+[a-zA-Z].*)|(n/a)", "")))
        df = df.withColumn('emp_length', trim(regexp_replace(df.emp_length, "< 1", "0")))
        df = df.withColumn('emp_length', trim(regexp_replace(df.emp_length, "10\\+", "10")).cast('float'))

        df = df.withColumn('verification_status',
                           trim(regexp_replace(df.verification_status, 'Source Verified', 'Verified')))

        df = df.withColumn('net', round(df.total_pymnt - df.loan_amnt, 2))
        return df

    def handle_cat_types(self, df):
        for col in df.columns:
            if df.dtypes[col] == 'object':
                df[col] = df[col].astype('category').cat.codes
            df[col] = df[col].fillna(0)
        return df

    def prepare_training_and_test_sets(self, df):
        X_train, X_test, Y_train, Y_test = train_test_split(df[predictors], df[target], test_size=0.3)
        return X_train, X_test, Y_train, Y_test

    def run(self):
        df = self.load_and_transform_data().toPandas()
        df = self.handle_cat_types(df)
        return self.prepare_training_and_test_sets(df)

    def load_and_transform_data_consumer(self):
        df = self.load_and_transform_data().toPandas()
        df = self.handle_cat_types(df)
        return self.spark.createDataFrame(df)
