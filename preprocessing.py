import pandas as pd
import numpy as np
import re
from rich import print
from sklearn.neighbors import LocalOutlierFactor

from logger import get_logger

logger = get_logger("logs/preprocessing.log")

class DataPreprocessing:

    def __init__(self, house_df: pd.DataFrame):
        self.__house_df = house_df
        self.__house_df = self.__replace_house_heating()
        self.__house_df = self.__convert_to_property_type()
        self.__house_df = self.__convert_to_bathroom_count()
        self.__house_df = self.__convert_gross_area_values_to_int()
        self.__house_df = self.__convert_net_area_values_to_int()
        self.__house_df = self.__convert_to_floor_number()
        self.__house_df = self.__convert_to_total_floors()
        self.__house_df = self.__convert_to_building_age()
        self.__house_df = self.__convert_room_and_livingroom_counts()
        self.__house_df = self.__split_house_location()
        self.__house_df = self.__convert_house_price_to_int()
        self.__house_df = self.__convert_furnished_values()
        self.__house_df = self.__convert_deed_status()
        self.__house_df = self.__convert_credit_eligible_values()
        self.__house_df = self.__drop_duplicate_records()
        self.__house_df = self.__remove_outliers_with_iqr()
        self.__house_df = self.__remove_outliers_with_lof()
        # self.__house_df = self.__remove_outliers_with_lof()
        

    def __convert_house_price_to_int(self) -> pd.DataFrame:

        if "house_price" not in self.__house_df.columns:
            logger.error("The 'house_price' column was not found in the DataFrame.")
            return self.__house_df

        self.__house_df  = self.__house_df[self.__house_df["house_price"]!="Bilgi yok"].reset_index(drop=True)

        try:
            self.__house_df["house_price"] =(
                                self.__house_df["house_price"]
                                .str.strip()
                                .str.replace("TL", "", regex=False)
                                .str.replace(".", "", regex=False)
                                .astype(int)
                                )
            
            self.__house_df["house_price"] = self.__house_df.loc[self.__house_df["house_price"] > 100_000, "house_price"]
            logger.info(f"House house_price values replaced successfully. {self.__house_df.shape}")
        except Exception as e:
            logger.error(f"Error converting house_price column: {e}", exc_info=True)

        return self.__house_df

    def __split_house_location(self) -> pd.DataFrame:

        if "house_location" not in self.__house_df.columns:
            logger.error("The 'house_location' column was not found in the DataFrame.")
            return self.__house_df

        self.__house_df = self.__house_df[self.__house_df["house_location"]!="Bilgi yok"].reset_index(drop=True)

        cleaned_data = []

        for index, value in enumerate(self.__house_df["house_location"]):
            try:
                if re.search(r"/", value):
                    data1 = re.split(r"/", value)
                    for _ in data1:
                        data1[-1] = data1[-1].strip()
                        data1[-1] = re.sub(r'Mah\.|Mh\.|Köyü|Bld\.', '', data1[-1])
                    cleaned_data.append(data1)
                    
                elif re.search(r"\n", value): 
                    data2 = re.split(r"\s+", value)
                
                    data2 = [element for element in data2 if "Mah." not in element]
                    
                    if len(data2) == 4:
                        data2[2] = data2[2] + " " + data2[3]
                        data2 = data2[:3]
                    
                    data2 = [re.sub(r'\(.*\)', '', element).strip() for element in data2]
        
                    cleaned_data.append(data2)
                        
                elif re.search(r"-", value):
                    data3 = re.split(r"-", value)
                    for _ in data3:
                        if re.search(r"Adana", data3[1]):
                            data3[1] = re.sub("Adana", "", data3[1]).strip()
                        data3[-1] = re.sub(r'\(.*\)', '', data3[-1]).strip()
                        data3[-1] = re.sub(r'Mahallesi|Köyü|Bld\.', '',  data3[-1])
                        data3[-1] = data3[-1].strip()
                    cleaned_data.append(data3)

                else : print(value)
            except Exception as e:
                logger.error(f"Error processing house_location at index {index}: {value} - Exception: {e}")
                cleaned_data.append([value])

        self.__house_df["province"] = [dc[0].strip() for dc in cleaned_data]
        self.__house_df["district"] = [dc[1].strip() for dc in cleaned_data]
        self.__house_df["neighborhood"] = [dc[-1].strip() for dc in cleaned_data]

        mask = self.__house_df["district"].map(self.__house_df["district"].value_counts()) >= 100
        self.__house_df = self.__house_df.loc[mask].reset_index(drop=True)

        self.__house_df["province"] = pd.Categorical(self.__house_df["province"].str.lower())
        self.__house_df["district"] = pd.Categorical(self.__house_df["district"].str.lower())
        self.__house_df["neighborhood"] = pd.Categorical(self.__house_df["neighborhood"].str.lower())

        self.__house_df["neighborhood"] = self.__house_df["neighborhood"].cat.remove_unused_categories()
        self.__house_df["district"] = self.__house_df["district"].cat.remove_unused_categories()

        self.__house_df = self.__house_df.drop(columns=["house_location"], axis="columns").reset_index(drop=True)

        logger.info("House house_location values replaced successfully.")

        return self.__house_df

    def __replace_house_heating(self) -> pd.DataFrame:
        
        if "heating" not in self.__house_df.columns:
            logger.error("The 'heating' column was not found in the DataFrame.")
            return self.__house_df
        
        self.__house_df = self.__house_df[(self.__house_df["heating"] != "Bilgi yok") & (self.__house_df["heating"] != "Group 3\n      Daha Fazla Filtre Göster")].reset_index(drop=True)

        values_to_replace = {
        
        'Kombi (Doğalgaz)': 'Kombi',
        'Kombi': 'Kombi',
        'Kombi Doğalgaz': 'Kombi',
        'Kombi Fueloil': 'Kombi',

        'Merkezi': 'Merkezi',
        'Merkezi (Pay Ölçer)': 'Merkezi',
        'Merkezi (Pay Öl...': 'Merkezi',
        'Merkezi Doğalgaz': 'Merkezi',
        'Merkezi Fueloil': 'Merkezi',
        'Merkezi Kömür': 'Merkezi',
        "Fancoil Ünitesi": "Merkezi",

        'Soba': 'Isıtma Yok',
        'Doğalgaz Sobası': 'Kat Kaloriferi',
        'Doğalgaz Sobalı': 'Kat Kaloriferi',
        'Sobalı': 'Isıtma Yok',

        'Yok': 'Isıtma Yok',
        'Isıtma Yok': 'Isıtma Yok',
        "Güneş Enerjisi": "Merkezi",

        'Var': 'Kombi',
        'Klima': 'Isıtma Yok',
        'Klimalı': 'Isıtma Yok',
        'Kat Kaloriferi': 'Kat Kaloriferi',
        'Yerden Isıtma': 'Yerden Isıtma',
        'Belirtilmemiş': 'Isıtma Yok',
        "Elektrikli Radyatör": "Isıtma Yok"
        }

        self.__house_df['heating'] = self.__house_df['heating'].replace(values_to_replace)
        self.__house_df["heating"] = pd.Categorical(self.__house_df["heating"])
        self.__house_df = self.__house_df.reset_index(drop=True)

        
        logger.info(f"House heating values replaced successfully. {self.__house_df.shape}")

        return self.__house_df

    def __convert_to_property_type(self) -> pd.DataFrame:

        if "property_type" not in self.__house_df.columns:
            logger.error("The 'property_type' column was not found in the DataFrame.")
            return self.__house_df
        
        self.__house_df = self.__house_df[(self.__house_df["property_type"] != "Bilgi yok") & 
                          (self.__house_df["property_type"] != "Villa")].reset_index(drop=True)

        self.__house_df["property_type"] = self.__house_df["property_type"].replace(["Daire"], "Satılık Daire")
        self.__house_df["property_type"] = pd.Categorical(self.__house_df["property_type"])
        self.__house_df["property_type"] = self.__house_df["property_type"].cat.remove_unused_categories()
        self.__house_df = self.__house_df.reset_index(drop=True)

        logger.info(f"House property_type values replaced successfully. {self.__house_df.shape}")

        return self.__house_df
        
    def __convert_to_bathroom_count(self)->pd.DataFrame:

        if "bathroom_count" not in self.__house_df.columns:
            logger.error("The 'bathroom_count' column was not found in the DataFrame.")
            return self.__house_df
        
        self.__house_df = self.__house_df[(self.__house_df["bathroom_count"]!="6+") & (self.__house_df["bathroom_count"]!="11") & (self.__house_df["bathroom_count"]!="12") & (self.__house_df["bathroom_count"]!="6") & (self.__house_df["bathroom_count"]!="8") & (self.__house_df["bathroom_count"]!="5") & (self.__house_df["bathroom_count"]!="9") & (self.__house_df["bathroom_count"]!="10")].reset_index(drop=True)

        self.__house_df["bathroom_count"] = (
                                self.__house_df["bathroom_count"]
                                .replace(["Bilgi yok", "Yok"], "0")
                                .astype(int)
                            )
        self.__house_df = self.__house_df.reset_index(drop=True)

        logger.info(f"House bathroom_count values replaced successfully. {self.__house_df.shape}")

        return self.__house_df

    def __convert_gross_area_values_to_int(self)-> pd.DataFrame:

        if "gross_area" not in self.__house_df.columns:
            logger.error("The 'gross_area' column was not found in the DataFrame.")
            return self.__house_df
        
        self.__house_df = self.__house_df[(self.__house_df["gross_area"]!="Bilgi yok")].reset_index(drop=True)

        cleaned_gross_area_values = []

        for value in self.__house_df["gross_area"]:
            if "m2" in value or "m²" in value: 
                value = re.sub(r'\s*m[²2]', '', value, flags=re.IGNORECASE).strip()
                if "." in value:
                    value = re.sub(r'\.', '', value).strip()

                cleaned_gross_area_values.append(int(value))
            elif "." in value:
                value = re.sub(r'\.', '', value).strip()
                cleaned_gross_area_values.append(int(value))
                
            else:cleaned_gross_area_values.append(int(value))

        self.__house_df["gross_area"] = cleaned_gross_area_values
        self.__house_df.loc[self.__house_df["gross_area"] > 1000, "gross_area"] = self.__house_df.loc[self.__house_df["gross_area"] > 1000, "gross_area"] //10

        self.__house_df = self.__house_df.reset_index(drop=True)


        logger.info(f"House gross_area values replaced successfully. {self.__house_df.shape}")

        return self.__house_df

    def __convert_net_area_values_to_int(self)-> pd.DataFrame:

        if "net_area" not in self.__house_df.columns:
            logger.error("The 'net_area' column was not found in the DataFrame.")
            return self.__house_df
        
        cleaned_net_area_values = []

        for value in self.__house_df["net_area"]:
            if "/" in value or "m2" in value or "m²" in value: 
                value = re.sub(r'\s*m[²2]|/', '', value, flags=re.IGNORECASE).strip()

                if "." in value:
                    value = re.sub(r'\.', '', value).strip()

                cleaned_net_area_values.append(int(value))
            elif "." in value:
                value = re.sub(r'\.', '', value).strip()
                cleaned_net_area_values.append(int(value))
                
            else:cleaned_net_area_values.append(int(value))


        self.__house_df["net_area"] = cleaned_net_area_values
        self.__house_df = self.__house_df.reset_index(drop=True)

        logger.info(f"House net_area values replaced successfully. {self.__house_df.shape}")

        

        return self.__house_df

    def __convert_to_floor_number(self) -> pd.DataFrame:

        if "floor_number" not in self.__house_df.columns:
            logger.error("The 'floor_number' column was not found in the DataFrame.")
            return self.__house_df

        self.__house_df = self.__house_df[(self.__house_df["floor_number"] != "Villa Tipi") & 
            (self.__house_df["floor_number"] != "Teras Katı") & 
            (self.__house_df["floor_number"] != "Müstakil") & 
            (self.__house_df["floor_number"] != "Çatı Dubleks") & (self.__house_df["floor_number"] != "Bahçe Dublex") & (self.__house_df["floor_number"] != "Villa Katı")]


        values_to_change = {
            "Giriş Altı Kot 3":-3,
            "Kot 3 (-3).Kat":-3,
            "Kot 2 (-2).Kat":-2,
            "Giriş Altı Kot 1":-1,
            "Bodrum":-1,
            "Bodrum Kat":-1,
            
            "Zemin":0,
            "Zemin Kat":0,
            "Giriş Katı":0,
            "Bahçe Katı":0,
            "Düz Giriş (Zemin)":0,

            "Yüksek Giriş":1,
            "Ara Kat":3,
            "En Üst Kat":4,
            "Çatı Katı":5,
            "Teras Katı":6
        }

        self.__house_df["floor_number"] = self.__house_df["floor_number"].replace(values_to_change)
        
        def other_convert(value):
            if isinstance(value, str):
                number_str = re.sub(r'\D', '', value)
                if number_str:
                    return int(number_str)
            return value

        self.__house_df["floor_number"] = self.__house_df["floor_number"].apply(other_convert)

        self.__house_df = self.__house_df.reset_index(drop=True)

        logger.info(f"House floor_number values replaced successfully. {self.__house_df.shape}")


        return self.__house_df
    
    def __convert_to_total_floors(self) -> pd.DataFrame:

        if "total_floors" not in self.__house_df.columns:
            logger.error("The 'total_floors' column was not found in the DataFrame.")
            return self.__house_df

        self.__house_df = self.__house_df[self.__house_df["total_floors"] != "Bilgi yok"]
        
        def convert(value):
            if isinstance(value, str):
                number_str = re.sub(r'\D', '', value)
                if number_str:
                    return int(number_str)
            return value

        self.__house_df["total_floors"] = self.__house_df["total_floors"].apply(convert)
      
        self.__house_df = self.__house_df.reset_index(drop=True)
        logger.info(f"House total_floors values replaced successfully. {self.__house_df.shape}")


        return self.__house_df

    def __convert_to_building_age(self) -> pd.DataFrame:

        if "building_age" not in self.__house_df.columns:
            logger.error("The 'building_age' column was not found in the DataFrame.")
            return self.__house_df

        self.__house_df = self.__house_df[(self.__house_df["building_age"] != "Bilgi yok") & 
        (self.__house_df["building_age"] != "11-15") & 
        (self.__house_df["building_age"] != "5-10") & 
        (self.__house_df["building_age"] != "16-20")]



        self.__house_df["building_age"] = self.__house_df["building_age"].str.replace("Sıfır Bina", "0")
        self.__house_df["building_age"] = self.__house_df["building_age"].str.extract(r"(\d+)")
        self.__house_df["building_age"] = self.__house_df["building_age"].astype(int)
        self.__house_df = self.__house_df.reset_index(drop=True)

        logger.info(f"House building_age values replaced successfully. {self.__house_df.shape}")

        return self.__house_df

    def __convert_room_and_livingroom_counts(self) -> pd.DataFrame:
    
        if "room_count" not in self.__house_df.columns:
            logger.error("The 'room_count' column was not found in the DataFrame.")
            return self.__house_df
        

        self.__house_df = self.__house_df[(self.__house_df["room_count"]!="Bilgi yok") & (self.__house_df["room_count"]!="Stüdyo")].reset_index(drop=True)

        
        room_count_list = []
        living_room_count = []
        
        for i, value in enumerate(self.__house_df["room_count"]):
            value = str(value).strip()
                    
            if '+' in value:
                parts = re.split(r'\s*\+\s*', value)
                if len(parts) == 2:
                    try:
                        oda = round(float(parts[0]))
                        salon = int(float(parts[1]))
                    except ValueError as e:
                        logger.error(f"Error converting parts for value '{value}': {e}")
                        oda, salon = np.nan, np.nan
                else:
                    logger.warning(f"Unexpected format for value with '+': {value}")
                    oda, salon = np.nan, np.nan
            else:
                match = re.search(r'(\d+(\.\d+)?)', value)
                if match:
                    try:
                        oda = round(float(match.group(1)))
                        salon = 0  
                    except ValueError as e:
                        logger.error(f"Error converting value '{value}': {e}")
                        oda, salon = np.nan, np.nan
                else:
                    logger.warning(f"No numeric value found in: {value}")
                    oda, salon = np.nan, np.nan
            
            room_count_list.append(oda)
            living_room_count.append(salon)
        
        self.__house_df["room_count"] = room_count_list
        self.__house_df["living_room_count"] = living_room_count
        self.__house_df = self.__house_df.reset_index(drop=True)
        logger.info(f"House room_count and living_room_count values replaced successfully. {self.__house_df.shape}")

        return self.__house_df

    def __drop_duplicate_records(self) -> pd.DataFrame:
        self.__house_df = self.__house_df.drop_duplicates(subset=['house_price', 'property_type', 'gross_area',
       'net_area', 'room_count', 'living_room_count', 'bathroom_count',
       'heating', 'floor_number', 'total_floors', 'building_age', 'province',
       'district', 'neighborhood', 'usage_status', 'deed_status', 'exchange',
       'credit_eligible', 'furnished',], keep='first')

        self.__house_df = self.__house_df.reset_index(drop=True)

        logger.info(f"Duplicate records dropped. {self.__house_df.shape}")
        return self.__house_df
    
    def __convert_furnished_values(self) -> pd.DataFrame:

        if "furnished" not in self.__house_df.columns:
            logger.error("The 'furnished' column was not found in the DataFrame.")
            return self.__house_df

        change_values = {
            "Eşyalı": "Eşyalı",
            "Eşyasız": "Eşyalı Değil",
            "Bilgi yok": "Eşyalı Değil",
            "Belirtilmemiş": "Eşyalı Değil",
            "Eşyalı Değil":"Eşyalı Değil",
            "Boş": "Eşyalı Değil",
        }

        self.__house_df["furnished"] = self.__house_df["furnished"].replace(change_values)

        self.__house_df = self.__house_df.reset_index(drop=True)

        self.__house_df["furnished"] = pd.Categorical(self.__house_df["furnished"])
        logger.info(f"House furnished values replaced successfully.")

        return self.__house_df
    
    def __convert_deed_status(self) -> pd.DataFrame:

        if "deed_status" not in self.__house_df.columns:
            logger.error("The 'deed_status' column was not found in the DataFrame.")
            return self.__house_df
        
        self.__house_df = self.__house_df[(self.__house_df["deed_status"] != "Bilgi yok") & (self.__house_df["deed_status"] != "Arsa") & (self.__house_df["deed_status"] != "Bilinmiyor") & (self.__house_df["deed_status"] != "Kooperatiften Tapu")]

        change_values = {
            "Kat Mülkiyeti": "Kat Mülkiyeti",
            "Kat İrtifakı": "Kat İrtifakı",
            "Yabancıdan": "Yabancıdan",

            "Tapu Yok": "Tapu Yok",
            "Tapu Kaydı Yok": "Tapu Yok",
            "Bilgi yok": "Tapu Yok",
            "Belirtilmemiş": "Tapu Yok",

            "Müstakil Tapulu": "Müstakil Tapu",
            "Hisseli Tapu": "Hisseli Tapu",
            "Arsa Tapulu": "Hisseli Tapu",
        }

        self.__house_df["deed_status"] = self.__house_df["deed_status"].replace(change_values)

        self.__house_df = self.__house_df.reset_index(drop=True)

        self.__house_df["deed_status"] = pd.Categorical(self.__house_df["deed_status"])
        logger.info(f"House deed_status values replaced successfully.")

        return self.__house_df  

    def __convert_credit_eligible_values(self) -> pd.DataFrame:


        if "credit_eligible" not in self.__house_df.columns:
            logger.error("The 'credit_eligible' column was not found in the DataFrame.")
            return self.__house_df
        
        self.__house_df = self.__house_df[(self.__house_df["credit_eligible"] != "Bilgi yok") & (self.__house_df["credit_eligible"] != "Belirtilmemiş") & (self.__house_df["credit_eligible"] != "Bilinmiyor")]

        change_values = {
            "Uygun" : "Krediye Uygun",
            "Krediye Uygun" : "Krediye Uygun",
            "Uygun Değil" : "Krediye Uygun Değil",
            "Uygun değil" : "Krediye Uygun Değil",
            "Krediye Uygun Değil" : "Krediye Uygun Değil",
        }

        self.__house_df["credit_eligible"] = self.__house_df["credit_eligible"].replace(change_values)

        self.__house_df = self.__house_df.reset_index(drop=True)

        self.__house_df["credit_eligible"] = pd.Categorical(self.__house_df["credit_eligible"])
        logger.info(f"House credit_eligible values replaced successfully.")

        return self.__house_df        

    def __remove_outliers_with_lof(self) -> pd.DataFrame:
            logger.info("Starting outlier removal using Local Outlier Factor.")
                
            numeric_df = self.__house_df.select_dtypes(include=["int64"])
                
            clf = LocalOutlierFactor(n_neighbors=50, contamination=0.05)
            clf.fit_predict(numeric_df)
            self.__house_df["NEF"] = clf.negative_outlier_factor_
            threshold = -1.15
            outliers = self.__house_df["NEF"] < threshold
            outliers_count = outliers.sum()
                
            self.__house_df = self.__house_df[~outliers]
            self.__house_df = self.__house_df.drop(columns=["NEF"], axis="columns").reset_index(drop=True)
                
            logger.info(f"Outlier removal completed. {outliers_count} outliers removed. {self.__house_df.shape}")
                
            return self.__house_df

    def __remove_outliers_with_iqr(self) -> pd.DataFrame:
        """
        Removes outliers using the IQR (Interquartile Range) method on a district-by-district basis.
        Uses different weights for each numerical column to adjust sensitivity.
        """
        logger.info("Starting district-based outlier removal using weighted IQR method.")

        # Select numerical columns
        numeric_columns = [
            'house_price', 'net_area', 'gross_area', 'room_count',
            'living_room_count', 'bathroom_count', 'building_age',
            'floor_number', 'total_floors'
        ]

        # Define IQR multiplier for each column (higher value = more tolerance)
        iqr_weights = {
            'house_price': 1.5, #ider tolerance for price
            'net_area': 1.5,         # Standard tolerance for net area
            'gross_area': 1.5,       # Standard tolerance for gross area
            'room_count': 1.5,       # Normal tolerance for room count
            'bathroom_count': 0.9,   # Normal tolerance for bathroom count
            'building_age': 1.0,     # Normal tolerance for building age
            'floor_number': 1.5,     # Normal tolerance for floor number
            'total_floors': 1.5,     # Normal tolerance for total floors
            'living_room_count': 2.0,  # Increased from 0.5 to 2.0 (more tolerance)
        }

        initial_count = len(self.__house_df)
        removed_per_district = {}
        removed_per_column = {}

        # Get all unique districts
        districts = self.__house_df['district'].unique()
        logger.info(f"Processing {len(districts)} unique districts for outlier removal.")

        # Process column order - process columns with less variation first
        column_order = ['living_room_count', 'bathroom_count', 'room_count', 'floor_number', 
                       'total_floors', 'building_age', 'net_area', 'gross_area', 'house_price']

        # Create a new DataFrame to store filtered results
        filtered_df = pd.DataFrame()

        # Process each district separately
        for district in districts:
            district_df = self.__house_df[self.__house_df['district'] == district].copy()
            district_initial_count = len(district_df)
            
            if district_initial_count < 20:  # Skip districts with too few records
                filtered_df = pd.concat([filtered_df, district_df])
                logger.info(f"District '{district}' has only {district_initial_count} records. Skipping outlier removal.")
                continue
                
            logger.info(f"Processing district: '{district}' with {district_initial_count} records")
            district_removed = 0
            
            for column in column_order:
                if column in district_df.columns and column in iqr_weights:
                    weight = iqr_weights[column]
                    column_initial = len(district_df)

                    # Calculate Q1, Q3, and IQR for this district's data
                    Q1 = district_df[column].quantile(0.25)
                    Q3 = district_df[column].quantile(0.75)
                    IQR = Q3 - Q1

                    # Calculate lower and upper bounds using weighted IQR
                    lower_bound = Q1 - weight * IQR
                    upper_bound = Q3 + weight * IQR

                    # Special handling for living_room_count
                    if column == 'living_room_count':
                        # Ensure living room count is between 0 and 2
                        lower_bound = 0
                        upper_bound = 2

                        # Log living_room_count distribution before filtering
                        value_counts = district_df[column].value_counts().sort_index()
                        logger.info(f"  - District '{district}', {column} distribution before: {dict(value_counts)}")

                    # Filter out outliers
                    district_df = district_df[
                        (district_df[column] >= lower_bound) & 
                        (district_df[column] <= upper_bound)
                    ]

                    # Log details about the filtering
                    column_removed = column_initial - len(district_df)
                    if column not in removed_per_column:
                        removed_per_column[column] = 0
                    removed_per_column[column] += column_removed
                    
                    if column_removed > 0:
                        logger.info(f"  - District '{district}', {column}: removed {column_removed} records (bounds: [{lower_bound:.2f}, {upper_bound:.2f}])")

            # Add the filtered district data to the result DataFrame
            filtered_df = pd.concat([filtered_df, district_df])
            
            district_removed = district_initial_count - len(district_df)
            removal_percentage = (district_removed / district_initial_count) * 100 if district_initial_count > 0 else 0
            removed_per_district[district] = district_removed
            
            logger.info(f"District '{district}': removed {district_removed} outliers ({removal_percentage:.2f}%)")

        # Final logging
        final_count = len(filtered_df)
        removed_count = initial_count - final_count
        removal_percentage = (removed_count / initial_count) * 100 if initial_count > 0 else 0
        
        logger.info(f"District-based IQR outlier removal completed.")
        logger.info(f"Initial total count: {initial_count}, Final count: {final_count}")
        logger.info(f"Removed {removed_count} outliers in total ({removal_percentage:.2f}%).")

        # Log per-column removal statistics
        for column, count in removed_per_column.items():
            percentage = (count / initial_count) * 100 if initial_count > 0 else 0
            logger.info(f"  - {column}: {count} records ({percentage:.2f}%)")

        # Ensure the filtered DataFrame has reset indices
        return filtered_df.reset_index(drop=True)

    def get_processing_df(self) -> pd.DataFrame:
        """
        İşlenmiş (preprocessed) DataFrame'i döndürür.
        Kullanılmayan kategorik değişkenleri temizler.
        """
        logger.info("Cleaning up unused categorical values before returning DataFrame")
        
        # Veri çerçevesinin kopyasını oluştur
        
        
        # Kategorik sütunları tespit et (Categorical veri tipinde olanlar)
        categorical_columns = self.__house_df.select_dtypes(include=['category']).columns.tolist()
        logger.info(f"Found {len(categorical_columns)} categorical columns: {categorical_columns}")
        
        # Her kategorik sütun için kullanılmayan kategorileri temizle
        for column in categorical_columns:
            # Kullanılmayan kategorileri say
            unused_before = len(self.__house_df[column].cat.categories) - len(self.__house_df[column].unique())
            
            if unused_before > 0:
                # Kullanılmayan kategorileri kaldır
                self.__house_df[column] = self.__house_df[column].cat.remove_unused_categories()
                
                # Kaldırılan kategori sayısını raporla
                unused_after = len(self.__house_df[column].cat.categories) - len(self.__house_df[column].unique())
                logger.info(f"Column '{column}': Removed {unused_before - unused_after} unused categories")
                logger.info(f"  - Categories before: {len(self.__house_df[column].cat.categories) + (unused_before - unused_after)}")
                logger.info(f"  - Categories after: {len(self.__house_df[column].cat.categories)}")
        
        # Temizlenmiş DataFrame'i döndür
        return self.__house_df

# Yeni bir main kontrol bloğu eklenerek son kısımdaki hatalı kod temizlendi.
if __name__ == "__main__":
    try:
        import pandas as pd
        house_df = pd.read_csv("data/all_data.csv", encoding="utf-8")
        processor = DataPreprocessing(house_df)
        final_df = processor.get_processing_df()

        print(final_df.columns)
        print(final_df.head())
        print(final_df.province.unique())
        print(final_df.neighborhood.unique())
        print(final_df.district.unique())
        print(final_df.property_type.unique())
        print(final_df.bathroom_count.unique())
        print(final_df.total_floors.unique())
        print(final_df.floor_number.unique())
        print(final_df.living_room_count.unique())
        print(final_df.room_count.unique())
        print(final_df.heating.unique())
        print(final_df.building_age.unique())
        print(final_df.house_price.describe())
        print(final_df["district"].value_counts())
        print(final_df["heating"].value_counts())
        print(final_df["neighborhood"].value_counts())
      # Group by district and get neighborhood counts

        # for i in final_df["neighborhood"].unique():
        #     print(i)
     
        logger.info("CSV file read successfully.")

        # Tüm mahalle sayılarını görmek için:
        # print("\n=== Tüm Mahalleler ve Kayıt Sayıları ===")
        
        # # Yöntem 1: Pandas'ın gösterim ayarlarını değiştirerek
        # print("\nYöntem 1: Pandas ayarlarını değiştirerek")
        # pd.set_option('display.max_rows', None)  # Tüm satırları göster
        # print(final_df["neighborhood"].value_counts())
        # pd.reset_option('display.max_rows')  # Ayarları sıfırla
        
        # # Yöntem 2: For döngüsü ile sıralı şekilde yazdırma
        # print("\nYöntem 2: Sıralı şekilde yazdırma")
        # neighborhood_counts = final_df["neighborhood"].value_counts()
        # for i, (neighborhood, count) in enumerate(neighborhood_counts.items(), 1):
        #     print(f"{i}. {neighborhood}: {count} adet")
        

        # final_df.to_csv("data/processed_data.csv", index=False, encoding="utf-8")
        
    except FileNotFoundError:
        logger.error("CSV file 'data/all_home_data.csv' not found.")