import sys
import argparse
import secrets
import hashlib
import sqlite3
from datetime import datetime, timedelta
import yaml
from pathlib import Path

# Fix python path for local imports
sys.path.append(str(Path(__file__).resolve().parent))
from core.database import Database

def get_db():
    try:
        with open("config_production.yaml", "r") as f:
            config = yaml.safe_load(f)
        db_path = config.get("database", {}).get("path", "securex_db.sqlite")
    except Exception:
        db_path = "securex_db.sqlite"
        
    db = Database(db_path)
    db.connect()
    db.initialize_schema()
    return db

def create_key(args):
    db = get_db()
    
    # Generate a secure random API key
    raw_key = f"sk_{secrets.token_urlsafe(32)}"
    
    # Hash the key for storage (we only store the hash)
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
    
    # Set expiration
    expires_at = None
    if args.days > 0:
        expires = datetime.utcnow() + timedelta(days=args.days)
        expires_at = expires.isoformat()
        
    key_id = db.create_api_key(key_hash, args.client, expires_at)
    
    if key_id:
        print("\n=== API KEY CREATED SUCCESSFULLY ===")
        print(f"Client: {args.client}")
        print(f"Key ID: {key_id}")
        print(f"API Key: {raw_key}")
        print("\nIMPORTANT: Keep this API key safe. It will not be shown again.")
        if expires_at:
            print(f"Expires: {expires_at} (UTC)")
        else:
            print("Expires: Never")
        print("====================================\n")
    else:
        print("Failed to create API key. Check logs.")

def list_keys(args):
    db = get_db()
    keys = db.get_all_api_keys()
    
    if not keys:
        print("No API keys found.")
        return
        
    print(f"\n{'ID':<5} | {'CLIENT':<25} | {'STATUS':<10} | {'VARS':<15} | {'EXPIRES'}")
    print("-" * 80)
    for key in keys:
        status = "Active" if key['is_active'] else "Revoked"
        expires = key['expires_at'] or "Never"
        if key['expires_at'] and key['is_active']:
            # Try to safely parse expiration
            try:
                expires_dt = datetime.fromisoformat(key['expires_at'].replace('Z', '+00:00'))
                if datetime.utcnow().timestamp() > expires_dt.timestamp():
                    status = "Expired"
            except:
                pass
                
        # show parts of the hash visually
        hash_preview = key['key_hash'][:8] + "..."
        print(f"{key['id']:<5} | {key['client_name']:<25} | {status:<10} | {hash_preview:<15} | {expires}")
    print("\n")

def revoke_key(args):
    db = get_db()
    if db.revoke_api_key(args.id):
        print(f"Successfully revoked API key ID: {args.id}")
    else:
        print("Failed to revoke API key.")

def main():
    parser = argparse.ArgumentParser(description="SecureX-Assist API Key Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.required = True
    
    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new API key")
    create_parser.add_argument("client", help="Name of the client application")
    create_parser.add_argument("--days", type=int, default=0, help="Days until expiration (0 = never expires)")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all API keys")
    
    # Revoke command
    revoke_parser = subparsers.add_parser("revoke", help="Revoke an API key")
    revoke_parser.add_argument("id", type=int, help="ID of the API key to revoke")
    
    args = parser.parse_args()
    
    if args.command == "create":
        create_key(args)
    elif args.command == "list":
        list_keys(args)
    elif args.command == "revoke":
        revoke_key(args)

if __name__ == "__main__":
    main()
