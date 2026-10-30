def get_update_data(db: str, init_data_perc: int):
    import pandas as pd
    
    tables = {
        'stats': [
            'badges', 'comments', 'posthistory', 'posts', 'postlinks', 'tags', 'users', 'votes' 
        ],
        'imdb': [
            "title", "movie_info", "movie_info_idx", "movie_companies", "movie_keyword", "cast_info"
        ]
    }[db]
    
    update_data = {}
    
    for table_name in tables:
        full_table_path = f"datasets/{db}/{table_name}.csv"
        initial_table_path = f"datasets/{db}/snapshots/{init_data_perc}/{table_name}.csv"
        
        df_init = pd.read_csv(initial_table_path, sep='|', na_values=['NULL'])
        df_full = pd.read_csv(full_table_path, sep='|', na_values=['NULL'])
        
        # Identify new rows in the full dataset that are not in the initial dataset
        merged = df_full.merge(df_init, how='outer', indicator=True)

        new_rows = merged[merged['_merge'] == 'left_only']

        df_new_rows = new_rows.drop(columns=['_merge'])

        update_data[table_name] = df_new_rows
        
    return update_data


def get_table_size_for_snapshot(db: str, perc: int):
    import pandas as pd
    
    tables = {
        'stats': [
            'badges', 'comments', 'posthistory', 'posts', 'postlinks', 'tags', 'users', 'votes' 
        ],
        'imdb': [
            "title", "movie_info", "movie_info_idx", "movie_companies", "movie_keyword", "cast_info"
        ]
    }[db]
    
    table_sizes = {}
    
    for table_name in tables:
        if perc == 100 or perc is None:
            snapshot_table_path = f"datasets/{db}/{table_name}.csv"
        else:
            snapshot_table_path = f"datasets/{db}/snapshots/{perc}/{table_name}.csv"
        
        df_snapshot = pd.read_csv(snapshot_table_path, sep='|', na_values=['NULL'])
        
        table_sizes[table_name] = len(df_snapshot)
    
    return table_sizes

