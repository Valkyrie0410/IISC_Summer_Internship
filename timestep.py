import sqlite3


def create_table_with_every_10th_record(db_path, original_table, new_table):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the new table
    cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {new_table} (
        frame_id INTEGER,
        distance_id INTEGER,
        distance REAL
    );""")

    cursor.execute("DELETE FROM pairwise_distances_new")

    for i in range(0, 100000, 2):  # Assuming your frame_ids range from 0 to 100000
        cursor.execute(f"""
        INSERT INTO {new_table} (frame_id, distance_id, distance)
        SELECT frame_id, distance_id, distance
        FROM {original_table}
        WHERE frame_id = ?;
        """, (i,))

    conn.commit()
    conn.close()
db_path = "/home/smart/Documents/IISC/sqlite_1.db"
original_table = 'pairwise_distances'
new_table = 'pairwise_distances_2'
create_table_with_every_10th_record(db_path, original_table, new_table)