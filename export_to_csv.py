import mysql.connector
import pandas as pd
import os

def export_data():
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="tictactoe"
        )
        print("Connected to MySQL.")
        
        # Export main dataset
        df_old = pd.read_sql("SELECT * FROM tic_tac_toc", conn)
        
        # Export user data if exists
        try:
            df_new = pd.read_sql("SELECT * FROM user_game_data", conn)
            if 'id' in df_new.columns: df_new = df_new.drop('id', axis=1)
            if 'created_at' in df_new.columns: df_new = df_new.drop('created_at', axis=1)
            df = pd.concat([df_old, df_new], ignore_index=True)
        except:
            df = df_old
            
        df.to_csv('tictactoe_data.csv', index=False)
        print("Successfully exported data to tictactoe_data.csv")
        conn.close()
    except Exception as e:
        print(f"Error exporting data: {e}")

if __name__ == "__main__":
    export_data()
