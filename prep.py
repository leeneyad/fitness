import pandas as pd


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns
                  .str.replace(r" \(.*\)", "", regex=True)
                  .str.replace("_", " ", regex=False)
                  .str.strip().str.lower()
                  .str.replace(" ", "_", regex=False))
    return df


def load_data(paths: dict) -> dict:
    """Load CSV files from given paths dict and return a mapping name->DataFrame.
    paths: {'df_gym_members': 'path/to/file.csv', ...}
    """
    out = {}
    for name, p in paths.items():
        try:
            out[name] = pd.read_csv(p)
        except FileNotFoundError:
            out[name] = None
    return out


def build_recommendation_df(df_megagym: pd.DataFrame) -> pd.DataFrame:
    df = df_megagym.copy()
    for col in ['type', 'body_part', 'equipment', 'level']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
        else:
            df[col] = ''
    df['features'] = df[['type', 'body_part', 'equipment', 'level']].astype(str).agg(' '.join, axis=1)
    return df


def clean_meals(df_meals: pd.DataFrame) -> pd.DataFrame:
    df = df_meals.copy()
    for col in ['calories', 'protein', 'fat', 'sodium']:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    df.drop(columns=[c for c in ['rating'] if c in df.columns], errors='ignore', inplace=True)
    # safe numeric conversion for tag columns
    tag_cols = df.columns[6:]
    for col in tag_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df[tag_cols] = df[tag_cols].fillna(0.0)
    # feature engineering
    if 'calories' in df.columns:
        df['calorie_category'] = pd.cut(df['calories'], bins=[0, 300, 600, df['calories'].max()], labels=['low_cal', 'moderate_cal', 'high_cal'], include_lowest=True).astype(str)
    return df
