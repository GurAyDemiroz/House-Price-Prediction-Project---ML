from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import os
from preprocessing import DataPreprocessing
from logger import get_logger

logger = get_logger("logs/preprocessing.log")

class DataPreprocessingPipeline:

    def __init__(self, house_df: pd.DataFrame):
        self.house_df = house_df
        self.pipeline = Pipeline(steps=[
            ('all_preprocessing', FunctionTransformer(self.__run_all_preprocessing_steps, validate=False)),
        ])
        self.__final_df = self.pipeline.fit_transform(self.house_df)
        logger.info("Data preprocessing completed successfully.")
        logger.info(f"Processed DataFrame shape: {self.__final_df.shape}")

    def __run_all_preprocessing_steps(self, house_df: pd.DataFrame) -> pd.DataFrame:
        processor = DataPreprocessing(house_df)
        return processor.get_processing_df()
    
    def get_final_df(self) -> pd.DataFrame:
        return self.__final_df



if __name__ == "__main__":
    try:
        house_df = pd.read_csv("data/all_data.csv")
        logger.info("CSV file read successfully.")
    except FileNotFoundError:
        logger.error("CSV file 'data/all_data.csv' not found.")
        exit(1)
    
    pipeline = DataPreprocessingPipeline(house_df)
    final_df = pipeline.get_final_df()

    print("\nYöntem 2: Sıralı şekilde yazdırma")
    neighborhood_counts = final_df["neighborhood"].value_counts()
    for i, (neighborhood, count) in enumerate(neighborhood_counts.items(), 1):
        print(f"{i}. {neighborhood}: {count} adet")
    
    
    logger.info("Pipeline processing completed. Processed DataFrame is ready.")
    logger.info(f"Processed DataFrame shape: {final_df.shape}")
    
    print(final_df.head())
    print(final_df.info())