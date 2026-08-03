"""
Face Enrollment Script
Enroll a user's face and store the embedding in the database.
"""

import sys
from pathlib import Path
import cv2

# Add project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.database import Database
from core.face_recognition_engine import FaceRecognitionEngine
from utils.helpers import load_config


def main():
    print("=" * 60)
    print("FACE ENROLLMENT - SecureX-Assist")
    print("=" * 60)

    config = load_config()

    db = Database(config["database"]["path"])
    db.connect()

    username = input("\nEnter username: ").strip()

    user = db.get_user_by_username(username)

    if not user:
        print(f"\nUser '{username}' not found.")
        db.close()
        return

    print("\nOpening camera...")

    face_engine = FaceRecognitionEngine()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open webcam.")
        db.close()
        return

    print("\nPress SPACE to capture your face.")
    print("Press ESC to cancel.\n")

    embedding = None

    while True:
        ret, frame = cap.read()

        if not ret:
            continue

        cv2.imshow("Face Enrollment", frame)

        key = cv2.waitKey(1)

        if key == 27:
            print("\nEnrollment cancelled.")
            cap.release()
            cv2.destroyAllWindows()
            db.close()
            return

        if key == 32:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            embedding = face_engine.enroll_face(rgb)

            if embedding is None:
                print("No face detected. Try again.")
                continue

            break

    cap.release()
    cv2.destroyAllWindows()

    print("\nSaving face profile...")

    db.deactivate_old_face_embeddings(user["id"])

try:
    embedding_id = db.store_face_embedding(
        user_id=user["id"],
        embedding=embedding,
        embedding_type="arcface",
        quality_score=1.0,
    )

    print("Returned embedding_id:", embedding_id)

except Exception as e:
    print("STORE ERROR:", e)
    raise

    db.close()


if __name__ == "__main__":
    main()