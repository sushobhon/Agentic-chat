import sqlite3
import json
import datetime

class SQLiteMemory:
    def __init__(self, db_path="memory.db"):
        self.conn = sqlite3.connect(db_path)
        self.cur = self.conn.cursor()
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT,
                content TEXT,
                timestamp TEXT,
                is_checkpoint BOOLEAN DEFAULT 0
            )
        """)
        self.conn.commit()

    def save(self, role, content):
        """
        Saves a single conversation entry to the database.
        
        Args:
            role (str): The role of the speaker (e.g., 'user', 'supervisor_agent').
            content (str): The text content of the message.
        """
        ts = datetime.datetime.now().isoformat()
        content_json = json.dumps(content)
        self.cur.execute("INSERT INTO memory (role, content, timestamp) VALUES (?, ?, ?)", 
                         (role, content_json, ts))
        self.conn.commit()

    def load_recent(self, limit=10):
        """
        Loads the most recent conversation entries from the database.
        
        Args:
            limit (int): The number of recent entries to retrieve.
            
        Returns:
            list: A list of tuples containing (role, content) of recent entries.
        """
        self.cur.execute("SELECT role, content FROM memory ORDER BY id DESC LIMIT ?", (limit,))
        return self.cur.fetchall()

    def save_checkpoint(self, role, content):
        """
        Saves a new entry and marks it as a checkpoint.
        This is useful for saving the state after a key decision or action.
        
        Args:
            role (str): The role of the agent making the checkpoint.
            content (str): The state or decision to be saved at the checkpoint.
            
        Returns:
            int: The ID of the newly created checkpoint entry.
        """
        ts = datetime.datetime.now().isoformat()
        self.cur.execute("INSERT INTO memory (role, content, timestamp, is_checkpoint) VALUES (?, ?, ?, ?)", 
                         (role, content, ts, 1))
        self.conn.commit()
        return self.cur.lastrowid

    def load_recent(self, limit=10):
        """
        Loads the most recent conversation entries from the database.
        
        Args:
            limit (int): The number of recent entries to retrieve.
            
        Returns:
            list: A list of tuples containing (role, content) of recent entries.
        """
        self.cur.execute("SELECT role, content FROM memory ORDER BY id DESC LIMIT ?", (limit,))
        return self.cur.fetchall()
    
    def load_from_checkpoint(self, checkpoint_id):
        """
        Loads all conversation entries up to and including a specific checkpoint.
        
        Args:
            checkpoint_id (int): The ID of the checkpoint to load from.
            
        Returns:
            list: A list of tuples containing (role, content) for all entries
                  up to the specified checkpoint. Returns an empty list if not found.
        """
        self.cur.execute("SELECT id FROM memory WHERE id = ? AND is_checkpoint = 1", (checkpoint_id,))
        if not self.cur.fetchone():
            print(f"Error: Checkpoint with ID {checkpoint_id} not found.")
            return []

        self.cur.execute("SELECT role, content FROM memory WHERE id <= ? ORDER BY id ASC", (checkpoint_id,))
        return self.cur.fetchall()

    def close(self):
        """
        Closes the database connection.
        """
        self.conn.close()

def convert_tuple_list_to_text(data):
    """
    Converts a list of tuples into a formatted text string.

    The function iterates through the list, and for each tuple:
    - It joins the elements with a colon ":".
    - It handles any complex data types (like nested strings with newlines)
      by ensuring they are properly formatted.
    - It joins the formatted tuple strings with a newline character "\n".

    Args:
        data (list): A list of tuples, where each tuple contains strings.

    Returns:
        str: A single formatted string.
    """
    formatted_parts = []
    for item in data:
        # Use a list comprehension to handle each element in the tuple
        # and ensure a valid string is created.
        # The json.loads() is used to correctly handle escaped characters from
        # the original string representation, like the newlines in your code.
        # We also strip any leading/trailing quotes from the strings.
        formatted_elements = []
        for element in item:
            try:
                # Attempt to deserialize from JSON to handle escaped characters
                # This is a robust way to handle the "CODE" and "OUTPUT" strings
                clean_element = json.loads(element)
            except (json.JSONDecodeError, TypeError):
                # If it's not JSON, just treat it as a regular string
                clean_element = str(element).strip('"')
            formatted_elements.append(clean_element)

        # Join the cleaned elements with a colon
        formatted_tuple = ": ".join(formatted_elements)
        formatted_parts.append(formatted_tuple)

    # Join all the formatted tuples with a newline
    return "\n".join(formatted_parts)