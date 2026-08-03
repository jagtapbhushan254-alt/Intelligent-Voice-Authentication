#!/usr/bin/env python3
"""
User Management Script for SecureX-Assist
Allows deleting users from the database
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.database import Database
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def list_users(db):
    """List all users"""
    try:
        users = db.get_all_users()
        
        if not users:
            print("No users found.")
            return []
        
        print("\nCurrent Users:")
        print("-" * 50)
        for user in users:
            account_type = user.get('account_type', 'standard_user')
            print(f"ID: {user['id']}, Username: {user['username']}, Type: {account_type}, Email: {user['email'] or 'N/A'}, Created: {user['created_at']}")
        print("-" * 50)
        return users
        
    except Exception as e:
        logger.error(f"Failed to list users: {e}")
        return []

def delete_user_by_id(db, user_id):
    """Delete user by ID"""
    try:
        success = db.delete_user(user_id)
        if success:
            print(f"[OK] User ID {user_id} deleted successfully.")
        else:
            print(f"[ERROR] Failed to delete user ID {user_id}.")
        return success
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        return False

def delete_all_users(db):
    """Delete all users from database"""
    try:
        users = db.get_all_users()
        
        if not users:
            print("No users to delete.")
            return True
        
        print(f"\n[WARNING] This will delete ALL {len(users)} users from the database!")
        print("This action CANNOT be undone!")
        confirm = input("Type 'DELETE ALL' to confirm: ").strip()
        
        if confirm != "DELETE ALL":
            print("Deletion cancelled.")
            return False
        
        # Delete each user
        success_count = 0
        fail_count = 0
        
        for user in users:
            if db.delete_user(user['id']):
                success_count += 1
                print(f"  ✓ Deleted user: {user['username']} (ID: {user['id']})")
            else:
                fail_count += 1
                print(f"  ✗ Failed to delete user: {user['username']} (ID: {user['id']})")
        
        print(f"\n[OK] Deleted {success_count} users successfully.")
        if fail_count > 0:
            print(f"[WARNING] Failed to delete {fail_count} users.")
        
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Error deleting all users: {e}")
        return False

def main():
    print("SecureX-Assist User Management")
    print("=" * 40)
    
    # Initialize database
    db_path = "securex_db.sqlite"
    if not os.path.exists(db_path):
        print(f"Database file '{db_path}' not found.")
        return
    
    db = Database(db_path)
    db.connect()
    db.initialize_schema()
    
    while True:
        print("\nOptions:")
        print("1. List all users")
        print("2. Delete user by ID")
        print("3. Delete ALL users")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            list_users(db)
            
        elif choice == "2":
            users = list_users(db)
            if users:
                try:
                    user_id = int(input("Enter user ID to delete: ").strip())
                    confirm = input(f"Are you sure you want to delete user ID {user_id}? (yes/no): ").strip().lower()
                    if confirm == "yes":
                        delete_user_by_id(db, user_id)
                    else:
                        print("Deletion cancelled.")
                except ValueError:
                    print("Invalid user ID.")
        
        elif choice == "3":
            users = list_users(db)
            if users:
                delete_all_users(db)
            
        elif choice == "4":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice.")
    
    db.close()

if __name__ == "__main__":
    main()