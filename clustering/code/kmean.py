import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
import ast
from pyproj import Transformer
import geopandas as gpd
import folium
import seaborn as sns

DATA_PATH = 'clustering/data/out/dataset_v202441617541'

def main():
    #Loading data
    data = pd.read_csv(DATA_PATH)
    #Removing outliers
    data = data[data['Building_heat_demand'] < 1000]
    data = data[data['Building_primary_fossil_energy'] < 1e4]
    data = data[data['volume'] > 0]
    data = data.dropna(subset=['Building_type'])

    #Removing NaN
    df = data.dropna()

    #Defining numerical energy labels
    energy_label_num = {'A+++++': 1, 'A++++': 2, 'A+++': 3, 'A++': 4, 'A+': 5, 'A': 6, 'B': 7, 'C': 8, 'D': 9, 'E': 10, 'F': 11, 'G': 12}
    #Create new label numerical column to make average label
    df['energy_label_num'] = df['Building_energy_class'].apply(lambda label: energy_label_num[label])

    #Aggregating residences at building level
    df = df.groupby('Building_bag_building_id').agg({
        'Building_bag_building_id': 'max',
        'energy_label_num': 'mean',  
        'Building_heat_demand': 'mean',
        'Building_share_renewable_energy': 'max',
        'Building_primary_fossil_energy': 'mean',
        'Building_energy_requirement': 'mean',
        'volume': 'mean',
        'year': 'mean',
        'volume': 'mean',
        'Building_usable_floor_area_thermal_zone': 'mean',
        'geographicalExtent': 'first'
    })[['Building_bag_building_id','energy_label_num', 'Building_heat_demand', 'Building_share_renewable_energy', 'Building_primary_fossil_energy', 
    'Building_energy_requirement', 'Building_usable_floor_area_thermal_zone', 'year', 'volume', 'geographicalExtent']]
    #Convert the energy_label_num back to categorical value
    num_to_energy_label = {v: k for k, v in energy_label_num.items()}
    df['energy_label'] = df['energy_label_num'].apply(lambda label: num_to_energy_label[round(label)])

    # Define numerical and categorical features
    numerical_features = ['Building_heat_demand', 'Building_primary_fossil_energy',
        'Building_energy_requirement',
        'Building_usable_floor_area_thermal_zone', 'volume', 'year'] #'Building_share_renewable_energy'
    categorical_features = ['energy_label']
    # Convert categorical columns to category dtype
    for col in categorical_features:
        df[col] = df[col].astype('category')

    # Preprocessing for numerical data
    numerical_transformer = StandardScaler()
    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', 'passthrough', categorical_features)
        ])
    # Apply preprocessing
    df_processed = preprocessor.fit_transform(df)
    # Convert processed data to a DataFrame (Optional, for PCA and visualization)
    df_processed = pd.DataFrame(df_processed)
    # Assuming df_processed is your preprocessed DataFrame and the optimal number of clusters is 3
    optimal_k = 5  # Replace with the optimal number of clusters you've determined from the elbow method
    # Create an instance of KPrototypes with the optimal number of clusters
    kproto = KPrototypes(n_clusters=optimal_k, init='Cao', n_init=1, verbose=1, random_state=42)
    # Fit the model and predict the clusters
    # Make sure to specify the correct indices for categorical features in your dataset
    cluster_assignment = kproto.fit_predict(df_processed, categorical=[6])
    # Assign the cluster labels to your original DataFrame
    df['Cluster'] = cluster_assignment

    # Define a transformer from EPSG:7415 to EPSG:4326 (WGS84)
    transformer = Transformer.from_crs("epsg:7415", "epsg:4326", always_xy=True)

    df['coordinates'] = df['geographicalExtent'].apply(lambda lat_long: transformer.transform(ast.literal_eval(lat_long)[0], ast.literal_eval(lat_long)[1]))
    df['latitude'] = df['coordinates'].apply(lambda coord: coord[1])
    df['longitude'] = df['coordinates'].apply(lambda coord: coord[0])

    # Convert DataFrame to GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326"  # Coordinate Reference System: WGS84
    )
    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world.plot(ax=ax, color='white', edgecolor='black')
    # Focus on Amsterdam area
    amsterdam_bounds = {
        'x_min': 4.85,  # min longitude
        'x_max': 4.95,  # max longitude
        'y_min': 52.35,  # min latitude
        'y_max': 52.40   # max latitude
    }
    # Define colors for each cluster
    colors = ['red', 'green', 'blue', 'yellow', 'purple']
    ax.set_xlim(4.84472, 4.96028)
    ax.set_ylim(52.33417, 52.39167)
    # Plot each cluster with different color
    for cluster, group in gdf.groupby('Cluster'):
        color = colors[cluster]
        group.plot(ax=ax, marker='o', markersize=3, label=f'Cluster {cluster}', color=color)
    plt.legend(title='Cluster')
    plt.show()

    results = df.groupby('Cluster')[['energy_label_num', 'Building_heat_demand',
        'Building_share_renewable_energy', 'Building_primary_fossil_energy',
        'Building_energy_requirement',
        'Building_usable_floor_area_thermal_zone', 'year', 'volume',
        'latitude', 'longitude']].describe()


    # Assuming you have loaded your DataFrame as `df`
    sampled_df = df
    # Setting the aesthetic style of the plots
    sns.set_theme(style="whitegrid")

    # Creating a list of numerical and categorical columns
    numerical_columns = numerical_features
    categorical_columns = categorical_features

    # Define colors for each cluster, consistent with the previous geographical plot
    num_clusters = sampled_df['Cluster'].nunique()
    palette = {i: colors[i % len(colors)] for i in range(num_clusters)}

    for column in numerical_columns:
        plt.figure(figsize=(10, 6))
        ax = sns.violinplot(x='Cluster', y=column,hue='Cluster', data=sampled_df, palette=palette)
        plt.title(f'Violin plot for {column} by Cluster')
        if column == 'volume':
            ax.set_ylim(-5000, 20000)  # Change 1000 to your desired maximum y-value
        elif column == 'Building_primary_fossil_energy':
            ax.set_ylim(-500, 2000)  # Change 1000 to your desired maximum y-value
        plt.show()
    # Plotting swarm plots for categorical data might not be directly insightful as swarm plots are typically used for numerical distributions.
    # However, if you want to visualize the distribution of categorical data, you can use count plots instead.
    for column in categorical_columns:
        print('categorcal column')
        plt.figure(figsize=(12, 8))
        sns.countplot(x=column, hue='Cluster', data=sampled_df, palette=palette)
        plt.title('Count of Energy Labels by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Count')
        plt.legend(title='Energy Label')
        plt.show()


    # Assuming you have loaded your DataFrame as `df`
    sampled_df = df
    # Setting the aesthetic style of the plots
    sns.set_theme(style="whitegrid")

    # Creating a list of numerical and categorical columns
    numerical_columns = numerical_features
    categorical_columns = categorical_features

    # Define colors for each cluster, consistent with the previous geographical plot
    num_clusters = sampled_df['Cluster'].nunique()
    palette = {i: colors[i % len(colors)] for i in range(num_clusters)}

    for column in numerical_columns:
        plt.figure(figsize=(10, 6))
        ax = sns.violinplot(x='Cluster', y=column,hue='Cluster', data=sampled_df, palette=palette)
        plt.title(f'Violin plot for {column} by Cluster')
        if column == 'volume':
            ax.set_ylim(-5000, 20000)  # Change 1000 to your desired maximum y-value
        elif column == 'Building_primary_fossil_energy':
            ax.set_ylim(-500, 2000)  # Change 1000 to your desired maximum y-value
        plt.show()
    # Plotting swarm plots for categorical data might not be directly insightful as swarm plots are typically used for numerical distributions.
    # However, if you want to visualize the distribution of categorical data, you can use count plots instead.
    for column in categorical_columns:
        print('categorcal column')
        plt.figure(figsize=(12, 8))
        sns.countplot(x=column, hue='Cluster', data=sampled_df, palette=palette)
        plt.title('Count of Energy Labels by Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Count')
        plt.legend(title='Energy Label')
        plt.show()

if __name__ == '__main__':
    main()

