import numpy as np
from database import get_all_embeddings
from face_engine import face_engine

# Recognition threshold
MATCH_THRESHOLD = 0.50


def search_face(image_file) -> dict:

    # Validate input
    if image_file is None:
        return {
            "match": False,
            "person": None,
            "similarity": 0.0,
            "message": "No image was provided."
        }

    # Generate embedding for the input face
    try:
        query_embedding = face_engine.get_embedding(image_file)

    except ValueError as error:
        return {
            "match": False,
            "person": None,
            "similarity": 0.0,
            "message": str(error)
        }

    except Exception as error:
        return {
            "match": False,
            "person": None,
            "similarity": 0.0,
            "message": f"Face processing failed: {error}"
        }

    # Get registered faces from database
    try:
        registered_people = get_all_embeddings()

    except Exception as error:
        return {
            "match": False,
            "person": None,
            "similarity": 0.0,
            "message": f"Database error: {error}"
        }

    # Check whether database is empty
    if not registered_people:
        return {
            "match": False,
            "person": None,
            "similarity": 0.0,
            "message": "No faces are registered in the database."
        }

    # Find the most similar registered face
    best_person = None
    best_similarity = -1.0

    for person in registered_people:

        stored_embedding = person["embedding"]

        try:
            similarity = face_engine.cosine_similarity(
                query_embedding,
                stored_embedding
            )

        except ValueError:
            # Skip invalid embeddings
            continue

        if similarity > best_similarity:
            best_similarity = similarity
            best_person = person

    # No valid embeddings found
    if best_person is None:
        return {
            "match": False,
            "person": None,
            "similarity": 0.0,
            "message": "No valid face embeddings found."
        }

    # Apply recognition threshold
    if best_similarity >= MATCH_THRESHOLD:

        person = {
            "id": best_person["id"],
            "name": best_person["name"],
            "person_id": best_person["person_id"],
            "created_at": best_person["created_at"]
        }

        return {
            "match": True,
            "person": person,
            "similarity": best_similarity,
            "message": "Face matched successfully."
        }

    # No match
    return {
        "match": False,
        "person": None,
        "similarity": best_similarity,
        "message": "No matching face found."
    }