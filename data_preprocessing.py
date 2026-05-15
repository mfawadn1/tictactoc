import mysql.connector
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data():
    """Connects to DB, loads data. Falls back to CSV if DB is unavailable."""
    df = None
    try:
        # 1. Try MySQL (Local/Secrets)
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="tictactoe"
        )
        print("Connected to MySQL database.")
        
        query_old = "SELECT * FROM tic_tac_toc"
        df_old = pd.read_sql(query_old, conn)
        
        try:
            query_new = "SELECT * FROM user_game_data"
            df_new = pd.read_sql(query_new, conn)
            if 'id' in df_new.columns: df_new = df_new.drop('id', axis=1)
            if 'created_at' in df_new.columns: df_new = df_new.drop('created_at', axis=1)
            df = pd.concat([df_old, df_new], ignore_index=True)
        except:
            df = df_old
        conn.close()
    except Exception as e:
        print(f"MySQL Connection failed: {e}")
        # 2. Fallback to CSV (for Streamlit Cloud)
        if os.path.exists('tictactoe_data.csv'):
            print("Loading data from tictactoe_data.csv...")
            df = pd.read_csv('tictactoe_data.csv')
        else:
            print("Error: No data source found (MySQL or CSV).")
            return None

    # Preprocessing
    try:
        # Decode bytes if necessary
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

        X = df.drop('Class', axis=1)
        y = df['Class']

        le = LabelEncoder()
        for col in X.columns:
            X[col] = le.fit_transform(X[col])
        y = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, le
    except Exception as e:
        print(f"Error in data preprocessing: {e}")
        return None

