import datetime
import hashlib
import sqlite3
import os
import re
from pathlib import Path
from typing import Union
from getpass import getpass

from spinesUtils.logging import Logger


class AuthenticationError(Exception):
    pass


class Authentication:
    """
    Authentication class for the LynseDB project

    The authentication is stored in a SQLite database file.
    """
    def __init__(self, path: str, logger: Logger):
        """
        Initialize the Authentication class.

        Parameters:
            path (str): The path to the database file.
            logger (Logger): The logger to use.
        """
        self._logger = logger
        self._path = Path(path).absolute() / "authentication.db"
        self._generate_table()
        self._alter_table_add_group()
        self._clean_duplicate_admin_records()

    def _generate_table(self):
        """
        Create the database file (if it doesn't exist) and generate the necessary table.
        """
        try:
            os.makedirs(self._path.parent, exist_ok=True)
            with sqlite3.connect(str(self._path)) as conn:
                c = conn.cursor()
                c.execute("""CREATE TABLE IF NOT EXISTS tokens
                             (username TEXT,
                              token TEXT,
                              role TEXT,
                              valid_until DATETIME,
                              salt TEXT,
                              group_name TEXT,
                              PRIMARY KEY (username, group_name))""")  # Composite primary key
            self._logger.info(f"Database file {self._path} has been created or accessed successfully")
        except sqlite3.OperationalError as e:
            self._logger.error(f"Failed to create or access database: {e}")
            raise
        except Exception as e:
            self._logger.error(f"Unexpected error occurred: {e}")
            raise

    def _alter_table_add_group(self):
        """
        Add the group_name column to the tokens table if it doesn't exist.
        """
        with sqlite3.connect(str(self._path)) as conn:
            c = conn.cursor()
            # Check if the group_name column already exists
            c.execute("PRAGMA table_info(tokens)")
            columns = [info[1] for info in c.fetchall()]
            if "group_name" not in columns:
                c.execute("ALTER TABLE tokens ADD COLUMN group_name TEXT DEFAULT 'DEFAULT_GROUP'")
                self._logger.info("Added 'group_name' column to tokens table with default 'DEFAULT_GROUP'")

    def _clean_duplicate_admin_records(self):
        """
        Clean duplicate admin records, keeping the latest one.
        """
        with sqlite3.connect(str(self._path)) as conn:
            c = conn.cursor()
            # Find duplicate admin records
            c.execute("""
                SELECT rowid, * FROM tokens WHERE username = 'admin'
            """)
            records = c.fetchall()
            if len(records) > 1:
                # Keep the last record and delete other records
                for record in records[:-1]:
                    rowid = record[0]
                    c.execute("DELETE FROM tokens WHERE rowid = ?", (rowid,))
                self._logger.info("Duplicate admin records have been cleaned up.")

    def _generate_token(self, username: str, passwd: str, role: str, valid_until: datetime.datetime, group_name: str) -> tuple:
        """
        Generate a unique token and salt value.
        """
        valid_until_str = valid_until.isoformat() if valid_until else "null"
        combined_data = f"{username}{passwd}{role}{valid_until_str}{group_name}"
        salt = os.urandom(16).hex()
        token = hashlib.sha256((combined_data + salt).encode()).hexdigest()
        return token, salt

    def _check_passwd(self, passwd: str) -> str:
        """
        Check passwords for compliance.
        """
        checks = [
            (len(passwd) >= 8, "Password must be at least 8 characters long"),
            (any(char.isupper() for char in passwd), "Must contain at least one uppercase letter"),
            (any(char.islower() for char in passwd), "Must contain at least one lowercase letter"),
            (any(char.isdigit() for char in passwd), "Must contain at least one digit"),
            (re.search(r'[!@#$%^&*(),.?":{}|<>]', passwd), "Must contain at least one special character"),
        ]
        return ', '.join(msg for valid, msg in checks if not valid)

    def _get_password_with_validation(self) -> Union[str, None]:
        """
        Get a new password from the user with validation.
        """
        time_to_try = 3
        while time_to_try > 0:
            new_passwd = getpass("Enter new password: ")
            warning_message = self._check_passwd(new_passwd)
            if warning_message:
                self._logger.info(warning_message)
                time_to_try -= 1
                if time_to_try == 0:
                    self._logger.info("Too many attempts.")
                    return None
            else:
                return new_passwd
        return None

    def _validate_username(self, username: str) -> bool:
        """
        Validate the username.
        """
        return (3 <= len(username) <= 20 and
                username.isalnum() and
                not username.isdigit() and
                username.lower() != 'admin')

    def _validate_group_name(self, group_name: str) -> bool:
        """
        Validate the group name to contain only uppercase letters.
        """
        return bool(re.fullmatch(r'[A-Z_0-9]+', group_name))

    def _prompt_for_old_password(self, username: str, group_name: str, old_passwd_token: str,
                                 old_salt: str, role: str, valid_until: datetime.datetime) -> Union[str, None]:
        """
        Prompt the user for the old password and validate it.
        """
        time_to_try = 3
        while time_to_try > 0:
            old_passwd = getpass("Please enter the old password: ")
            valid_until_str = valid_until.isoformat() if valid_until else "null"
            combined_data = f"{username}{old_passwd}{role}{valid_until_str}{group_name}"
            given_old_passwd_token = hashlib.sha256((combined_data + old_salt).encode()).hexdigest()

            if old_passwd_token != given_old_passwd_token:
                self._logger.info("Incorrect old password.")
                time_to_try -= 1
                if time_to_try == 0:
                    self._logger.info("Too many attempts.")
                    return None
            else:
                return old_passwd
        return None

    def _confirm_new_password(self, new_passwd: str) -> bool:
        """
        Confirm that the new password matches.
        """
        time_to_try = 3
        while time_to_try > 0:
            valid_new_passwd = getpass("Enter new password again: ")
            if new_passwd != valid_new_passwd:
                self._logger.info("New passwords do not match.")
                time_to_try -= 1
                if time_to_try == 0:
                    self._logger.info("Too many attempts.")
                    return False
            else:
                return True
        return False

    def create_admin_account(self) -> Union[str, None]:
        """
        Create or reset an admin token and save it to the db table.
        """
        # Fetch the latest token and salt from the database before any operations
        with sqlite3.connect(self._path) as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM tokens WHERE username = 'admin'")
            result = c.fetchone()

        old_passwd = None
        role = 'admin'
        valid_until = None

        if result:
            old_passwd_token, old_salt = result[1], result[4]
            old_passwd = self._prompt_for_old_password('admin', 'ALL_GROUPS', old_passwd_token, old_salt, role, valid_until)
            if old_passwd is None:
                return None

        new_passwd = self._get_password_with_validation()
        if new_passwd is None:
            return None

        if old_passwd and new_passwd == old_passwd:
            self._logger.info("New password cannot be the same as the old password.")
            return None

        if not self._confirm_new_password(new_passwd):
            return None

        # Generate new token and salt
        token, salt = self._generate_token('admin', new_passwd, 'admin', valid_until, 'ALL_GROUPS')

        with sqlite3.connect(self._path) as conn:
            c = conn.cursor()
            c.execute("""INSERT OR REPLACE INTO tokens
                         (username, token, role, valid_until, salt, group_name)
                         VALUES (?, ?, ?, ?, ?, ?)""",
                      ('admin', token, 'admin', None, salt, 'ALL_GROUPS'))  # the admin can access all groups

        self._logger.info("Admin account created or updated")
        return token

    def create_user_account(self) -> Union[str, None]:
        """
        Create or reset a user token and save it to the db table.

        Every user must have a unique username within the same group. Usernames cannot be 'admin'.
        The user valid time is 30 days.
        """
        # Check if admin exists
        with sqlite3.connect(self._path) as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM tokens WHERE username = 'admin'")
            admin_exists = c.fetchone()[0] > 0
        if not admin_exists:
            self._logger.info("Cannot create user account without an admin account.")
            return None

        # let user choose group name
        time_to_try = 3
        while time_to_try > 0:
            group_input = input("Enter group name (press Enter for 'DEFAULT_GROUP'): ").strip()
            group_name = group_input if group_input else "DEFAULT_GROUP"
            if self._validate_group_name(group_name):
                break
            else:
                self._logger.info("Invalid group name. Only uppercase letters and '_' and numbers are allowed.")
                time_to_try -= 1
                if time_to_try == 0:
                    self._logger.info("Too many attempts.")
                    return None

        username = input("Enter username: ").strip()
        while not self._validate_username(username):
            self._logger.info("Invalid username. Use 3-20 characters (letters and digits, not all digits) and not 'admin'.")
            username = input("Enter username: ").strip()

        # # Check if username exists in the group
        with sqlite3.connect(self._path) as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM tokens WHERE username = ? AND group_name = ?", (username, group_name))
            result = c.fetchone()

        # Fetch existing user details if any
        if result:
            old_passwd_token, old_valid_until_str, old_salt, old_group = result[1], result[3], result[4], result[5]
            role = 'user'
            valid_until = datetime.datetime.fromisoformat(old_valid_until_str) if old_valid_until_str else None

            old_passwd = self._prompt_for_old_password(username, group_name, old_passwd_token,
                                                       old_salt, role, valid_until)
            if old_passwd is None:
                return None
        else:
            role = 'user'
            valid_until = datetime.datetime.now() + datetime.timedelta(days=30)

        new_passwd = self._get_password_with_validation()
        if new_passwd is None:
            return None

        if result and new_passwd == old_passwd:
            self._logger.info("New password cannot be the same as the old password.")
            return None

        if not self._confirm_new_password(new_passwd):
            return None

        if not result:
            valid_until = datetime.datetime.now() + datetime.timedelta(days=30)

        token, salt = self._generate_token(username, new_passwd, 'user', valid_until, group_name)

        with sqlite3.connect(self._path) as conn:
            c = conn.cursor()
            c.execute("""INSERT OR REPLACE INTO tokens
                         (username, token, role, valid_until, salt, group_name)
                         VALUES (?, ?, ?, ?, ?, ?)""",
                      (username, token, 'user', valid_until.isoformat(), salt, group_name))

        self._logger.info(f"User account '{username}' created or updated in group '{group_name}'")
        return token

    def delete_user_account(self) -> bool:
        """
        Delete a user token from the db table.
        """
        admin_token, admin_salt = self._get_admin_token()
        if admin_token is None or admin_salt is None:
            self._logger.info("Admin account not found.")
            return False

        role = 'admin'

        time_to_try = 3
        while time_to_try > 0:
            valid_admin_passwd = getpass("Enter admin password: ")
            combined_data = f"admin{valid_admin_passwd}{role}{'null'}ALL_GROUPS"
            given_admin_passwd_token = hashlib.sha256((combined_data + admin_salt).encode()).hexdigest()

            if admin_token != given_admin_passwd_token:
                self._logger.info("Incorrect admin password.")
                time_to_try -= 1
                if time_to_try == 0:
                    self._logger.info("Too many attempts.")
                    return False
            else:
                break

        group_name = input("Enter group name of the user to delete: ").strip()
        username = input("Enter username to delete: ").strip()
        if not self._validate_group_name(group_name):
            self._logger.info("Invalid group name. Only uppercase letters and '_' and numbers are allowed.")
            return False

        if username.lower() == 'admin':
            self._logger.info("Cannot delete the admin account.")
            return False

        with sqlite3.connect(self._path) as conn:
            c = conn.cursor()
            c.execute("DELETE FROM tokens WHERE username = ? AND group_name = ? AND username != 'admin'", (username, group_name))
            if c.rowcount == 0:
                self._logger.warning(f"User '{username}' in group '{group_name}' not found or attempt to delete admin account")
                return False

        self._logger.info(f"User account '{username}' in group '{group_name}' deleted")
        return True

    def reset_user_valid_time(self) -> bool:
        """
        Reset the valid time of a user token. Default is 30 days.
        """
        admin_token, admin_salt = self._get_admin_token()
        if admin_token is None or admin_salt is None:
            self._logger.info("Admin account not found.")
            return False

        role = 'admin'

        time_to_try = 3
        while time_to_try > 0:
            valid_admin_passwd = getpass("Enter admin password: ")
            combined_data = f"admin{valid_admin_passwd}{role}{'null'}ALL_GROUPS"
            given_admin_passwd_token = hashlib.sha256((combined_data + admin_salt).encode()).hexdigest()

            if admin_token != given_admin_passwd_token:
                self._logger.info("Incorrect admin password.")
                time_to_try -= 1
                if time_to_try == 0:
                    self._logger.info("Too many attempts.")
                    return False
            else:
                break

        group_input = input("Enter group name (press Enter for 'DEFAULT_GROUP'): ").strip()
        group_name = group_input if group_input else "DEFAULT_GROUP"
        if not self._validate_group_name(group_name):
            self._logger.info("Invalid group name. Only uppercase letters and '_' and numbers are allowed.")
            return False

        username = input("Enter username: ").strip()

        new_valid_until = datetime.datetime.now() + datetime.timedelta(days=30)

        with sqlite3.connect(self._path) as conn:
            c = conn.cursor()
            c.execute("UPDATE tokens SET valid_until = ? WHERE username = ? AND group_name = ? AND role = 'user'",
                      (new_valid_until.isoformat(), username, group_name))
            if c.rowcount == 0:
                self._logger.warning(f"User '{username}' in group '{group_name}' not found or attempt to modify admin account")
                return False

        self._logger.info(f"Valid time for user '{username}' in group '{group_name}' updated to {new_valid_until}")
        return True

    def redef_user_valid_time(self, username: str, group_name: str, valid_until: datetime.datetime) -> bool:
        """
        Redefine the valid time of a user token.
        """
        admin_token, admin_salt = self._get_admin_token()
        if admin_token is None or admin_salt is None:
            self._logger.info("Admin account not found.")
            return False

        role = 'admin'

        time_to_try = 3
        while time_to_try > 0:
            valid_admin_passwd = getpass("Enter admin password: ")
            combined_data = f"admin{valid_admin_passwd}{role}{'null'}ALL_GROUPS"
            given_admin_passwd_token = hashlib.sha256((combined_data + admin_salt).encode()).hexdigest()

            if admin_token != given_admin_passwd_token:
                self._logger.info("Incorrect admin password.")
                time_to_try -= 1
                if time_to_try == 0:
                    self._logger.info("Too many attempts.")
                    return False
            else:
                break

        with sqlite3.connect(self._path) as conn:
            c = conn.cursor()
            c.execute("UPDATE tokens SET valid_until = ? WHERE username = ? AND group_name = ? AND role = 'user'",
                      (valid_until.isoformat(), username, group_name))
            if c.rowcount == 0:
                self._logger.warning(f"User '{username}' in group '{group_name}' not found or attempt to modify admin account")
                return False

        self._logger.info(f"Valid time for user '{username}' in group '{group_name}' updated to {valid_until}")
        return True

    def _get_admin_token(self) -> Union[tuple[str, str], tuple[None, None]]:
        """
        Get the admin token and salt from the database.
        """
        with sqlite3.connect(self._path) as conn:
            c = conn.cursor()
            c.execute("SELECT token, salt FROM tokens WHERE username = 'admin'")
            result = c.fetchone()
            return result if result else (None, None)

    def _get_admin_salt(self) -> str:
        """
        Get the admin salt from the database.
        """
        with sqlite3.connect(self._path) as conn:
            c = conn.cursor()
            c.execute("SELECT salt FROM tokens WHERE username = 'admin'")
            result = c.fetchone()
            return result[0] if result else ''

    def is_token_valid(self, token: str) -> bool:
        """
        Check if a token is valid.
        """
        with sqlite3.connect(self._path, check_same_thread=False) as conn:
            c = conn.cursor()
            c.execute("SELECT valid_until FROM tokens WHERE token = ?", (token,))
            result = c.fetchone()
            if result:
                valid_until_str = result[0]
                if valid_until_str:
                    valid_until = datetime.datetime.fromisoformat(valid_until_str)
                else:
                    valid_until = None

                if valid_until is None or valid_until > datetime.datetime.now():
                    return True
        return False

    def get_all_users(self) -> list:
        """
        Get all users from the database along with their group names.
        """
        admin_token, admin_salt = self._get_admin_token()
        if admin_token is None or admin_salt is None:
            self._logger.info("Admin account not found.")
            return []

        role = 'admin'

        time_to_try = 3
        while time_to_try > 0:
            valid_admin_passwd = getpass("Enter admin password: ")
            combined_data = f"admin{valid_admin_passwd}{role}{'null'}ALL_GROUPS"
            given_admin_passwd_token = hashlib.sha256((combined_data + admin_salt).encode()).hexdigest()

            if admin_token != given_admin_passwd_token:
                self._logger.info("Incorrect admin password.")
                time_to_try -= 1
                if time_to_try == 0:
                    self._logger.info("Too many attempts.")
                    return []
            else:
                break

        with sqlite3.connect(self._path) as conn:
            c = conn.cursor()
            c.execute("SELECT username, role, valid_until, group_name, token FROM tokens WHERE role = 'user'")
            result = c.fetchall()

            if not result:
                self._logger.info("No users found.")
                return []

            return [{
                "username": user[0],
                "role": user[1],
                "valid_until": datetime.datetime.fromisoformat(user[2]).strftime("%Y-%m-%d %H:%M:%S.%f") if user[2] else None,
                "group_name": user[3],
                "is_valid": self.is_token_valid(user[4])
            } for user in result]

    def get_group_users(self, group_name: str = "DEFAULT_GROUP") -> list:
        """
        Get users from a specified group. Defaults to 'DEFAULT_GROUP'.
        """
        if not self._validate_group_name(group_name):
            self._logger.info("Invalid group name. Only uppercase letters and '_' and numbers are allowed.")
            return []

        admin_token, admin_salt = self._get_admin_token()
        if admin_token is None or admin_salt is None:
            self._logger.info("Admin account not found.")
            return []

        role = 'admin'

        time_to_try = 3
        while time_to_try > 0:
            valid_admin_passwd = getpass("Enter admin password: ")
            combined_data = f"admin{valid_admin_passwd}{role}{'null'}ALL_GROUPS"
            given_admin_passwd_token = hashlib.sha256((combined_data + admin_salt).encode()).hexdigest()

            if admin_token != given_admin_passwd_token:
                self._logger.info("Incorrect admin password.")
                time_to_try -= 1
                if time_to_try == 0:
                    self._logger.info("Too many attempts.")
                    return []
            else:
                break

        with sqlite3.connect(self._path) as conn:
            c = conn.cursor()
            c.execute("SELECT username, role, valid_until, group_name, token FROM tokens WHERE role = 'user' AND group_name = ?", (group_name,))
            result = c.fetchall()

            if not result:
                self._logger.info(f"No users found in group '{group_name}'.")
                return []

            return [{
                "username": user[0],
                "role": user[1],
                "valid_until": datetime.datetime.fromisoformat(user[2]).strftime("%Y-%m-%d %H:%M:%S.%f") if user[2] else None,
                "group_name": user[3],
                "is_valid": self.is_token_valid(user[4])
            } for user in result]
